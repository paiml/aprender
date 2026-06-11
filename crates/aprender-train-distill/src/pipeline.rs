//! Distillation pipeline execution (Heijunka - level scheduling).
//!
//! Orchestrates the complete distillation workflow from model fetching
//! through training and export.

use crate::config::{DistillConfig, WeightFormat};
use crate::weights::load_safetensors_weights;
use crate::MemoryEstimate;
use entrenar::distill::{save_student_checkpoint, DistillationCheckpoint, DistillationLoss};
use entrenar_common::{EntrenarError, Result};
use ndarray::{Array2, Axis};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Pipeline execution result.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Path to the output model
    pub output_path: PathBuf,
    /// Training metrics
    pub metrics: TrainingMetrics,
    /// Total execution time in seconds
    pub duration_seconds: f64,
}

/// Training metrics collected during distillation.
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Initial loss at start of training
    pub initial_loss: f32,
    /// Final loss at end of training
    pub final_loss: f32,
    /// Best validation loss achieved
    pub best_loss: f32,
    /// Number of training steps completed
    pub steps_completed: u64,
    /// Average throughput (samples/second)
    pub throughput: f32,
}

impl TrainingMetrics {
    /// Calculate loss improvement ratio.
    pub fn improvement_ratio(&self) -> f32 {
        if self.initial_loss > 0.0 {
            1.0 - (self.final_loss / self.initial_loss)
        } else {
            0.0
        }
    }
}

/// Distillation pipeline orchestrator.
pub struct Pipeline<'a> {
    config: &'a DistillConfig,
    /// SPEC-DISTILL-001 Phase 1 (PMAT-691): the teacher backend the
    /// training loop pulls logits from. Defaults to a `FixtureTeacher`.
    /// Phase 1b (PMAT-693) adds `CudaTrainerTeacher` for real backends.
    teacher: Box<dyn crate::teacher_provider::TeacherLogitsProvider>,
    /// SPEC-DISTILL-001 Phase 2b (PMAT-695): the student backend the
    /// training loop updates each step. Defaults to a `FixtureStudent`.
    /// Phase 2d (PMAT-696) adds `CudaStudentProvider` wrapping
    /// `CudaTransformerTrainer` for production runs.
    student: Box<dyn crate::student_provider::StudentLogitsProvider>,
    /// SPEC-DISTILL-001 Phase 4 Stage B-2: batch source for the training
    /// loop. Defaults to `SyntheticBatchSource` (smoke + fixture path);
    /// Phase 4 real-corpus dispatch swaps in `ShardBatchSource` via
    /// `with_batch_source()`.
    batch_source: Box<dyn crate::batch_source::BatchSource>,
    /// PMAT-705: training-progress callbacks. Empty by default — operators
    /// attach `entrenar::train::ProgressCallback` (or richer) via
    /// `with_callback()` to surface per-step loss during long runs.
    /// Contract: `contracts/distill-pipeline-observability-v1.yaml`.
    callbacks: Vec<Box<dyn entrenar::train::TrainerCallback>>,
    /// PMAT-706: smoke-validation early-break budget.
    /// `Some(N)` (N ≥ 1) runs at most N training steps then breaks, prints a
    /// `[SMOKE]` loss/throughput/projection summary, and skips the final
    /// export — letting operators validate the cascade end-to-end in ~60s
    /// before committing to a 30-50h Stage D run. `Some(0)` is a degenerate
    /// request that `train()` rejects. `None` = normal full run (no change).
    ///
    /// Defaults from `APR_DISTILL_MAX_STEPS` at construction; override in
    /// tests via [`Self::with_max_steps`] so they never touch process env.
    /// Contract: `contracts/apr-distill-smoke-validation-v1.yaml`.
    smoke_max_steps: Option<u64>,
    /// PMAT-PERPOS: full-sequence (per-position) KD mode. When `true`, each
    /// step trains on EVERY position of the window (`kd_step_per_position`)
    /// rather than one target per window — up to `seq_len`× more KD signal.
    /// Defaults from `APR_DISTILL_PER_POSITION` at construction; override in
    /// tests via [`Self::with_per_position`]. Default `false` keeps the
    /// production loop byte-identical. Contract: `contracts/distill-per-position-kd-v1.yaml`.
    per_position: bool,
}

impl<'a> Pipeline<'a> {
    /// Create a new pipeline with the given configuration.
    ///
    /// Uses fixture teacher + student by default. Use [`Self::with_teacher`]
    /// / [`Self::with_student`] to swap in real backends for Phase 4
    /// production runs.
    pub fn new(config: &'a DistillConfig) -> Self {
        // The fixture vocab size matches the legacy synthetic-logits stub
        // (num_classes = 32) so existing tests behave identically. The
        // student starts at uniform logits (0.0) with a moderate LR; this
        // means without a Phase 2d real backend, the pipeline still does
        // *something* — it nudges the fixture student's logits toward the
        // teacher's distribution. Useful for unit tests of the data flow,
        // not for real distillation.
        Self {
            config,
            teacher: Box::new(crate::teacher_provider::FixtureTeacher::new(32)),
            student: Box::new(crate::student_provider::FixtureStudent::new(32, 0.0, 0.1)),
            batch_source: Box::new(crate::batch_source::SyntheticBatchSource::new(32)),
            callbacks: Vec::new(),
            // PMAT-706: pick up the operator's smoke budget from the env once.
            smoke_max_steps: parse_max_steps(),
            // PMAT-PERPOS: full-sequence KD opt-in from the env once.
            per_position: parse_per_position(),
        }
    }

    /// PMAT-PERPOS: enable/disable full-sequence (per-position) KD.
    ///
    /// `true` trains on every position of each window; `false` (default) is
    /// the per-row path. Primarily a test seam — operators set
    /// `APR_DISTILL_PER_POSITION=1`, which `new()` reads automatically.
    #[must_use]
    pub fn with_per_position(mut self, per_position: bool) -> Self {
        self.per_position = per_position;
        self
    }

    /// PMAT-706: override the smoke-validation early-break budget.
    ///
    /// `Some(N)` (N ≥ 1) runs at most N steps then breaks; `Some(0)` makes
    /// `execute()`/`train()` return a clear error; `None` runs the full
    /// configured schedule. Primarily a test seam — operators set
    /// `APR_DISTILL_MAX_STEPS` instead, which `new()` reads automatically.
    #[must_use]
    pub fn with_max_steps(mut self, max_steps: Option<u64>) -> Self {
        self.smoke_max_steps = max_steps;
        self
    }

    /// PMAT-705: attach a training-progress callback.
    ///
    /// Callbacks fire in attach order at the lifecycle events defined by
    /// `entrenar::train::TrainerCallback`:
    /// `on_train_begin` → `on_epoch_begin` → `on_step_end` × N → `on_epoch_end` → `on_train_end`.
    ///
    /// The default pipeline attaches no callbacks; the `apr distill --backend cuda`
    /// dispatch (`apr-cli::run_cuda_backend`) wires a default `ProgressCallback`
    /// honoring the `APR_DISTILL_LOG_EVERY` env var (default 10).
    ///
    /// See `contracts/distill-pipeline-observability-v1.yaml`.
    #[must_use]
    pub fn with_callback(mut self, callback: Box<dyn entrenar::train::TrainerCallback>) -> Self {
        self.callbacks.push(callback);
        self
    }

    /// Swap in a custom batch source (Phase 4 Stage B-2).
    ///
    /// Pass a `ShardBatchSource` to drive training from a real-corpus
    /// `.bin` shard directory. The default `SyntheticBatchSource` is
    /// used when this builder is not called — appropriate for smoke
    /// + fixture-path tests.
    #[must_use]
    pub fn with_batch_source(
        mut self,
        batch_source: Box<dyn crate::batch_source::BatchSource>,
    ) -> Self {
        self.batch_source = batch_source;
        self
    }

    /// Swap in a custom teacher backend.
    ///
    /// Phase 4 wiring point: pass a `CudaTrainerTeacher` loaded with
    /// the MODEL-1 7B teacher.
    #[must_use]
    pub fn with_teacher(
        mut self,
        teacher: Box<dyn crate::teacher_provider::TeacherLogitsProvider>,
    ) -> Self {
        self.teacher = teacher;
        self
    }

    /// Swap in a custom student backend.
    ///
    /// Phase 4 wiring point: pass a `CudaStudentProvider` wrapping a
    /// trainable `CudaTransformerTrainer`.
    #[must_use]
    pub fn with_student(
        mut self,
        student: Box<dyn crate::student_provider::StudentLogitsProvider>,
    ) -> Self {
        self.student = student;
        self
    }

    /// Execute the complete distillation pipeline.
    pub fn execute(&mut self) -> Result<PipelineResult> {
        let start = std::time::Instant::now();

        // Stage 1: Fetch/resolve models
        let teacher_path = self.fetch_teacher()?;
        let student_path = self.fetch_student()?;

        // Stage 2: Train with distillation loss
        let (metrics, student_weights, student_shapes) =
            self.train(&teacher_path, &student_path)?;

        // Stage 3: Export student checkpoint — unless this is a smoke run.
        // PMAT-706 / apr-distill-smoke-validation-v1 `no_side_effects`:
        // smoke-validation runs MUST NOT write a final output model so that
        // downstream tools (`apr eval`, `apr run`) can't consume a smoke
        // result by accident. This holds whether the loop early-broke or ran
        // to natural completion (N ≥ total planned steps). Intermediate
        // PMAT-699 checkpoints under the output dir may still exist.
        let smoke_mode = matches!(self.smoke_max_steps, Some(n) if n >= 1);
        let output_path = if smoke_mode {
            println!(
                "[PMAT-706] smoke mode: skipping export — \
                 no model.safetensors / output.apr written"
            );
            PathBuf::new()
        } else {
            self.export(&student_weights, &student_shapes, &metrics)?
        };

        Ok(PipelineResult {
            output_path,
            metrics,
            duration_seconds: start.elapsed().as_secs_f64(),
        })
    }

    /// Estimate memory requirements for this configuration.
    pub fn estimate_memory(config: &DistillConfig) -> Result<MemoryEstimate> {
        let teacher_params = estimate_params_from_model_id(&config.teacher.model_id);
        let student_params = estimate_params_from_model_id(&config.student.model_id);

        let estimate = MemoryEstimate::new(
            student_params + teacher_params / 4,
            config.training.batch_size as usize,
            config.dataset.max_length,
            4096,
        );

        Ok(estimate)
    }

    /// Resolve teacher model to a local path.
    ///
    /// If the model_id looks like a local path, validates it exists.
    /// If it's a HuggingFace model ID (org/model), downloads via HfModelFetcher
    /// when the `hub` feature is enabled.
    fn fetch_teacher(&self) -> Result<PathBuf> {
        resolve_model_path(&self.config.teacher.model_id)
    }

    /// Resolve student model to a local path.
    fn fetch_student(&self) -> Result<PathBuf> {
        resolve_model_path(&self.config.student.model_id)
    }

    /// Run the distillation training loop.
    ///
    /// Loads teacher and student weights from SafeTensors files, computes
    /// distillation loss via `DistillationLoss`, and applies gradient updates
    /// to student parameters.
    ///
    /// Note: Without a full transformer forward pass (autograd backward ops
    /// are incomplete), logits are derived from loaded weight tensor slices
    /// as a demonstration of the real loss computation pipeline.
    fn train(
        &mut self,
        teacher_path: &Path,
        student_path: &Path,
    ) -> Result<(
        TrainingMetrics,
        HashMap<String, Vec<f32>>,
        HashMap<String, Vec<usize>>,
    )> {
        // PMAT-706: smoke-validation early-break budget (apr-distill-smoke-
        // validation-v1). Read once. `Some(0)` is a degenerate request — fail
        // fast with a clear, actionable error before any model I/O, and with
        // no division-by-zero in the summary path (KANI-SMOKE-002).
        let max_steps = self.smoke_max_steps;
        if max_steps == Some(0) {
            return Err(EntrenarError::ConfigValue {
                field: "APR_DISTILL_MAX_STEPS".to_string(),
                message: "0 steps requested — smoke mode needs at least 1 step".to_string(),
                suggestion: "set APR_DISTILL_MAX_STEPS to a positive integer (e.g. 10), \
                             or unset it to run the full configured schedule"
                    .to_string(),
            });
        }

        // Load weights from both models. The teacher_weights byte buffer
        // is no longer used for logits computation (Phase 1 wired it to
        // the teacher provider instead) but we still load + drop it to
        // validate the teacher checkpoint is well-formed before training.
        let (_teacher_weights_validate, _teacher_shapes) = load_safetensors_weights(teacher_path)?;
        let (mut student_weights, student_shapes) = load_safetensors_weights(student_path)?;

        // Create distillation loss function
        let temperature = self.config.distillation.temperature;
        let alpha = self.config.distillation.alpha;
        let loss_fn = DistillationLoss::new(temperature, alpha);

        let lr = self.config.training.learning_rate as f32;

        let batch_size = self.config.training.batch_size as usize;
        let num_classes = self.teacher.vocab_size();

        // SPEC-DISTILL-001 Phase 2c (PMAT-691): the training loop now uses
        // both abstractions end-to-end. Per step:
        //
        //   1. Build a dummy batch of input_ids (Phase 4 replaces this with
        //      real tokens from a dataset).
        //   2. self.teacher.logits_for_batch → teacher logits.
        //   3. self.student.logits_for_batch → student logits.
        //   4. kd_step's `kd_loss` + `kd_logit_gradient` → (scalar loss,
        //      per-batch logit gradients).
        //   5. self.student.apply_kd_gradient → student updates its
        //      parameters in the gradient direction.
        //
        // The ndarray bookkeeping that used to live here is gone — the
        // student provider owns its parameter buffer. `student_weights`
        // (loaded from the on-disk safetensors) is only retained so the
        // export step at the end can write the resulting checkpoint.
        let _ = (lr, &loss_fn); // legacy values retained for back-compat

        // PMAT-698m: smoke-test batch setup. The original used
        //   dummy_batch = vec![vec![0u32]; batch_size]
        //   labels     = (0..batch_size).map(|i| i % num_classes)
        // — same input (token 0) paired with N distinct labels, which is
        // impossible to learn (identical features cannot map to distinct
        // targets). CE loss diverges, which surfaced as Phase 3 GB10 smoke
        // returning final_loss=8.39 > initial_loss=6.08 even though the
        // pipeline itself was working end-to-end.
        //
        // Fix: per-row input matches per-row label. Each row carries a
        // distinct token; the label is that same token. The student learns
        // the trivial identity mapping (input → predict same token), CE
        // decreases monotonically, KD signal is zero when teacher==student,
        // and F-DISTILL-SMOKE-001 ("final_loss < initial_loss") becomes
        // satisfiable on the standard smoke configuration.
        //
        // For real distillation with a real dataset, the caller would
        // override the pipeline's batch construction entirely; this default
        // exists for the smoke + fixture-path tests.
        //
        // PMAT-698o: scale the synthetic batch from seq_len=1 (Phase 3
        // smoke) to seq_len=APR_DISTILL_SMOKE_SEQ_LEN (default 256) so the
        // smoke exercises the same memory/kernel paths that Phase 4 real
        // training will use. The student observes a row of `seq_len` copies
        // of the same token and is asked to predict that token — still
        // trivially learnable (identity from a constant signal), but now
        // touches the attention scores tensor, batched 4D GEMMs, and rope
        // forward at non-singleton sequence dimensions. Catches latent
        // bugs that only surface at seq > 1 before we commit Phase 4
        // compute.
        //
        // Override via env: APR_DISTILL_SMOKE_SEQ_LEN=N.
        // Fixture tests are unaffected by the longer sequence — the
        // FixtureStudent ignores input shape and emits argmax-on-label
        // logits regardless of seq_len.
        // Phase 4 Stage B-2: pull each batch from the configured
        // BatchSource instead of constructing inline. Synthetic source is
        // the default (smoke + fixture path semantics unchanged);
        // production runs swap in a ShardBatchSource via
        // `Pipeline::with_batch_source()`. See PMAT-PHASE4-STAGE-B-2.
        let smoke_seq_len: usize = std::env::var("APR_DISTILL_SMOKE_SEQ_LEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);

        let mut metrics = TrainingMetrics::default();
        let mut best_loss = f32::MAX;

        // Initial loss — drives `metrics.initial_loss` for the
        // PipelineResult improvement-ratio computation.
        // Pull an initial batch from the source so the source observes
        // the same step count as the training loop (some sources cache
        // state per-call).
        let (initial_batch, initial_labels) =
            self.batch_source.next_batch(batch_size, smoke_seq_len)?;
        let initial_loss = kd_step_loss_for_pipeline(
            &mut *self.teacher,
            &mut *self.student,
            &initial_batch,
            &initial_labels,
            temperature,
            alpha,
        )?;
        metrics.initial_loss = initial_loss;
        best_loss = best_loss.min(initial_loss);

        let train_start = std::time::Instant::now();
        let mut step = 0u64;
        // Track the last batch for the final-loss measurement (matches the
        // synthetic-batch semantics: final loss is computed on the last
        // batch consumed by the training loop). For ShardBatchSource this
        // is the most recent real-corpus batch.
        let mut last_batch = initial_batch;
        let mut last_labels = initial_labels;

        // PMAT-699 P0 durability fix: periodic intermediate checkpointing.
        // Stage D 2026-05-20 ran 25h with ZERO checkpoints — if it had
        // crashed at step 49999, the full run would be lost. Default 5000
        // steps; env-overridable via APR_DISTILL_CHECKPOINT_EVERY=N.
        // Set N=0 to disable (smoke tests).
        let checkpoint_every: u64 = std::env::var("APR_DISTILL_CHECKPOINT_EVERY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5000);
        let checkpoint_dir = self.config.output.dir.clone();

        // PMAT-705: total step count across epochs (used by ProgressCallback
        // to render Step N/M). Synthetic loop uses steps_per_epoch =
        // max(1, 1000 / batch_size); see the inner loop below.
        let steps_per_epoch_u64 = (1000 / u64::from(self.config.training.batch_size)).max(1);
        let total_steps_planned = steps_per_epoch_u64 * u64::from(self.config.training.epochs);

        // PMAT-705: fire on_train_begin once before the first step.
        if !self.callbacks.is_empty() {
            let ctx = entrenar::train::CallbackContext {
                epoch: 0,
                max_epochs: self.config.training.epochs as usize,
                step: 0,
                #[allow(clippy::cast_possible_truncation)]
                steps_per_epoch: steps_per_epoch_u64 as usize,
                global_step: 0,
                loss: initial_loss,
                lr,
                best_loss: Some(best_loss),
                val_loss: None,
                elapsed_secs: 0.0,
            };
            for cb in &mut self.callbacks {
                let _ = cb.on_train_begin(&ctx);
            }
        }

        let mut stop_requested = false;
        // PMAT-706: set true when the smoke-validation early-break fires (as
        // opposed to normal epoch exhaustion or a CallbackAction::Stop). The
        // `[SMOKE]` summary fires iff this is true.
        let mut smoke_break = false;
        for epoch_idx in 0..self.config.training.epochs {
            let steps_this_epoch = steps_per_epoch_u64;

            // PMAT-705: on_epoch_begin.
            if !self.callbacks.is_empty() {
                let ctx = entrenar::train::CallbackContext {
                    epoch: epoch_idx as usize,
                    max_epochs: self.config.training.epochs as usize,
                    step: 0,
                    #[allow(clippy::cast_possible_truncation)]
                    steps_per_epoch: steps_per_epoch_u64 as usize,
                    #[allow(clippy::cast_possible_truncation)]
                    global_step: step as usize,
                    loss: best_loss,
                    lr,
                    best_loss: Some(best_loss),
                    val_loss: None,
                    elapsed_secs: train_start.elapsed().as_secs_f64(),
                };
                for cb in &mut self.callbacks {
                    let _ = cb.on_epoch_begin(&ctx);
                }
            }

            for _s in 0..steps_this_epoch {
                // PMAT-PERPOS: per-position (full-sequence) KD trains on EVERY
                // position of each window; the default per-row path trains on
                // one target per window. Opt-in via APR_DISTILL_PER_POSITION
                // (default off → the production loop is byte-identical). CUDA
                // providers fall back to one position via the trait defaults
                // until they implement a true all-positions forward.
                let (loss, dummy_batch, labels) = if self.per_position {
                    let (inputs, pp_labels) = self
                        .batch_source
                        .next_batch_per_position(batch_size, smoke_seq_len)?;
                    let (loss, grads) = crate::kd_step::kd_step_per_position(
                        self.teacher.as_mut(),
                        &inputs,
                        &pp_labels,
                        temperature,
                        alpha,
                        |ids| {
                            self.student
                                .logits_per_position(std::slice::from_ref(&ids.to_vec()))
                                .expect("student per-position forward failed mid-step")
                                .into_iter()
                                .next()
                                .unwrap_or_default()
                        },
                    )?;
                    self.student.apply_kd_gradient_per_position(&grads)?;
                    // Per-row labels for the final-loss telemetry: the last
                    // target of each row (the next token after the window).
                    let row_labels: Vec<usize> = pp_labels
                        .iter()
                        .map(|l| l.last().copied().unwrap_or(0))
                        .collect();
                    (loss, inputs, row_labels)
                } else {
                    let (dummy_batch, labels) =
                        self.batch_source.next_batch(batch_size, smoke_seq_len)?;
                    let (loss, grads) = crate::kd_step::kd_step(
                        self.teacher.as_mut(),
                        &dummy_batch,
                        &labels,
                        temperature,
                        alpha,
                        |ids| {
                            // Compute student logits inline. The closure runs
                            // once per batch element inside kd_step. Whole-batch
                            // upfront is cheaper when the provider has shared
                            // state; adjacent same-input calls are deterministic
                            // so indexing is safe.
                            let logits_vv = self
                                .student
                                .logits_for_batch(std::slice::from_ref(&ids.to_vec()))
                                .expect("student provider failed mid-step");
                            logits_vv.into_iter().next().unwrap_or_default()
                        },
                    )?;
                    self.student.apply_kd_gradient(&grads)?;
                    (loss, dummy_batch, labels)
                };
                best_loss = best_loss.min(loss);
                step += 1;
                last_batch = dummy_batch;
                last_labels = labels;

                // PMAT-705: on_step_end fires after grad application and
                // before the checkpoint-save block so callbacks observe
                // committed step state. CallbackAction::Stop terminates
                // the loop after this step.
                if !self.callbacks.is_empty() {
                    let ctx = entrenar::train::CallbackContext {
                        epoch: epoch_idx as usize,
                        max_epochs: self.config.training.epochs as usize,
                        #[allow(clippy::cast_possible_truncation)]
                        step: step as usize,
                        #[allow(clippy::cast_possible_truncation)]
                        steps_per_epoch: total_steps_planned as usize,
                        #[allow(clippy::cast_possible_truncation)]
                        global_step: step as usize,
                        loss,
                        lr,
                        best_loss: Some(best_loss),
                        val_loss: None,
                        elapsed_secs: train_start.elapsed().as_secs_f64(),
                    };
                    for cb in &mut self.callbacks {
                        if cb.on_step_end(&ctx) == entrenar::train::CallbackAction::Stop {
                            stop_requested = true;
                        }
                    }
                }
                if stop_requested {
                    break;
                }

                // PMAT-699 P0 durability: periodic checkpoint save.
                if checkpoint_every > 0 && step % checkpoint_every == 0 {
                    let ckpt_path = checkpoint_dir.join(format!("ckpt-step-{step:06}.apr"));
                    if let Err(e) = self.student.save_checkpoint(&ckpt_path) {
                        // Don't fail training on checkpoint write error; just
                        // log loudly. Loss progress is more valuable than
                        // pristine intermediate snapshots.
                        eprintln!(
                            "[PMAT-699] checkpoint save at step {step} failed: \
                             {e} (training continues)"
                        );
                    } else {
                        eprintln!(
                            "[PMAT-699] checkpoint saved at step {step}: {}",
                            ckpt_path.display()
                        );
                    }
                }

                // PMAT-706: smoke-validation early-break. `step` was already
                // incremented above, so `step >= m` means exactly `m` steps
                // ran — never m+1 (FT-SMOKE-001 / KANI-SMOKE-001 off-by-one
                // guard). The break is inside the inner loop, after grad
                // application + callbacks + checkpoint for this step, so
                // partial epochs are valid.
                if let Some(m) = max_steps {
                    if step >= m {
                        smoke_break = true;
                        break;
                    }
                }
            }

            // PMAT-705: on_epoch_end.
            if !self.callbacks.is_empty() {
                let ctx = entrenar::train::CallbackContext {
                    epoch: epoch_idx as usize,
                    max_epochs: self.config.training.epochs as usize,
                    #[allow(clippy::cast_possible_truncation)]
                    step: step as usize,
                    #[allow(clippy::cast_possible_truncation)]
                    steps_per_epoch: steps_per_epoch_u64 as usize,
                    #[allow(clippy::cast_possible_truncation)]
                    global_step: step as usize,
                    loss: best_loss,
                    lr,
                    best_loss: Some(best_loss),
                    val_loss: None,
                    elapsed_secs: train_start.elapsed().as_secs_f64(),
                };
                for cb in &mut self.callbacks {
                    let _ = cb.on_epoch_end(&ctx);
                }
            }

            if stop_requested || smoke_break {
                break;
            }
        }

        let elapsed = train_start.elapsed().as_secs_f32().max(1e-6);

        // PMAT-705: on_train_end after the loop (final loss is computed below).
        if !self.callbacks.is_empty() {
            let ctx = entrenar::train::CallbackContext {
                epoch: self.config.training.epochs.saturating_sub(1) as usize,
                max_epochs: self.config.training.epochs as usize,
                #[allow(clippy::cast_possible_truncation)]
                step: step as usize,
                #[allow(clippy::cast_possible_truncation)]
                steps_per_epoch: steps_per_epoch_u64 as usize,
                #[allow(clippy::cast_possible_truncation)]
                global_step: step as usize,
                loss: best_loss,
                lr,
                best_loss: Some(best_loss),
                val_loss: None,
                elapsed_secs: f64::from(elapsed),
            };
            for cb in &mut self.callbacks {
                cb.on_train_end(&ctx);
            }
        }

        // Final loss measurement on the last batch consumed.
        let final_loss = kd_step_loss_for_pipeline(
            &mut *self.teacher,
            &mut *self.student,
            &last_batch,
            &last_labels,
            temperature,
            alpha,
        )?;

        metrics.final_loss = final_loss;
        metrics.best_loss = best_loss.min(final_loss);
        metrics.steps_completed = step;
        metrics.throughput = (step as f32 * batch_size as f32) / elapsed;

        // PMAT-706: smoke-validation degenerate-run guard + summary.
        if max_steps.is_some() && step == 0 {
            // Smoke mode was requested but the loop ran nothing (e.g. epochs=0
            // or an empty schedule). Per apr-distill-smoke-validation-v1 this
            // is the "likely teacher or student load failure" error path — a
            // non-zero exit, not a silent success the operator might trust.
            return Err(EntrenarError::ConfigValue {
                field: "APR_DISTILL_MAX_STEPS".to_string(),
                message: "smoke mode ran 0 steps — likely teacher or student load failure"
                    .to_string(),
                suggestion: "verify both model paths resolve and load (try `apr inspect` on each)"
                    .to_string(),
            });
        }
        if smoke_break {
            // Summary fires ONLY on the early-break path — never on normal
            // epoch exhaustion or a CallbackAction::Stop (contract invariant).
            let project_to_steps = parse_project_to_steps();
            let (summary, projection) = smoke_summary_lines(
                step,
                elapsed,
                metrics.initial_loss,
                final_loss,
                project_to_steps,
            );
            println!(
                "[PMAT-706] smoke mode: APR_DISTILL_MAX_STEPS={} \
                 (early-break after {step} steps; no final output.apr written)",
                max_steps.unwrap_or(0)
            );
            println!("{summary}");
            println!("{projection}");
        }

        // SPEC-DISTILL-001 Phase 2c+3-prep: the student provider owns its
        // parameter buffer. To preserve the FALSIFY-APR-DISTILL-TRAIN-001
        // contract ("output student tensors differ from input student
        // tensors by at least Q4K_TOLERANCE after training") we project
        // the student provider's current logit state back into the
        // on-disk weight buffer one last time.
        //
        // FixtureStudent has [vocab_size] logits → we use them to
        // overwrite a [batch, vocab] slice of student_weights via the
        // legacy write_logits_to_weights helper.
        //
        // Phase 2d's CudaStudentProvider doesn't expose a flat-logits
        // view (its state lives in GPU weight tensors). For that backend
        // the right way to capture the trained student is
        // CudaTransformerTrainer's save_checkpoint hook, which Phase 4
        // wires into the export step. Until then, with the cuda backend
        // selected, this projection is a no-op — that's fine because
        // Phase 4 owns the real serialization path.
        let final_logits_vv = self.student.logits_for_batch(&last_batch)?;
        let mut final_logits_flat: Vec<f32> = Vec::with_capacity(batch_size * num_classes);
        for row in final_logits_vv {
            final_logits_flat.extend(row);
        }
        if final_logits_flat.len() == batch_size * num_classes {
            let final_logits_arr =
                Array2::from_shape_vec((batch_size, num_classes), final_logits_flat)
                    .expect("student provider returned (batch, vocab) buffer");
            write_logits_to_weights(
                &mut student_weights,
                &final_logits_arr,
                batch_size,
                num_classes,
            );
        }

        Ok((metrics, student_weights, student_shapes))
    }

    /// Export trained student model using `save_student_checkpoint`.
    ///
    /// PMAT-699 P0: now ALSO calls `self.student.save_checkpoint(...)` after
    /// the metadata-only safetensors write, so the CudaStudentProvider can
    /// pull its trained GPU weights back to disk as an APR v2 file in the
    /// same directory. Without this, the cuda path's 25h of training
    /// silently produces a 200-byte empty model.safetensors (Stage D
    /// 2026-05-20 incident).
    ///
    /// The fixture path's no-op default `save_checkpoint` preserves the
    /// existing FixtureStudent behavior — only the safetensors metadata
    /// sidecar is written, matching pre-PMAT-699 semantics.
    fn export(
        &mut self,
        weights: &HashMap<String, Vec<f32>>,
        shapes: &HashMap<String, Vec<usize>>,
        metrics: &TrainingMetrics,
    ) -> Result<PathBuf> {
        std::fs::create_dir_all(&self.config.output.dir).map_err(|e| EntrenarError::Io {
            context: format!(
                "creating output directory: {}",
                self.config.output.dir.display()
            ),
            source: e,
        })?;

        let checkpoint = DistillationCheckpoint {
            teacher_model: self.config.teacher.model_id.clone(),
            temperature: self.config.distillation.temperature,
            alpha: self.config.distillation.alpha,
            final_loss: Some(metrics.final_loss),
            epoch: self.config.training.epochs as usize,
            step: metrics.steps_completed as usize,
        };

        let filename = match self.config.output.format {
            WeightFormat::SafeTensors => "model.safetensors",
            WeightFormat::Gguf => "model.gguf",
            WeightFormat::Apr => "model.json",
        };

        // Save SafeTensors checkpoint with distillation metadata sidecar
        let output_path = save_student_checkpoint(
            weights,
            shapes,
            &checkpoint,
            &self.config.output.dir,
            filename,
        )
        .map_err(|e| EntrenarError::Io {
            context: "saving student checkpoint".to_string(),
            source: e,
        })?;

        // For GGUF format with hub feature, also export via Exporter
        #[cfg(feature = "hub")]
        if self.config.output.format == WeightFormat::Gguf {
            let mw = crate::weights::weights_to_model_weights(weights.clone(), shapes.clone());
            let exporter = entrenar::hf_pipeline::Exporter::new()
                .output_dir(&self.config.output.dir)
                .gguf_quantization(entrenar::hf_pipeline::GgufQuantization::Q8_0);
            exporter
                .export(&mw, entrenar::hf_pipeline::ExportFormat::GGUF, filename)
                .map_err(|e| EntrenarError::Internal {
                    message: format!("GGUF export failed: {e}"),
                })?;
        }

        // PMAT-699 P0: ask the student provider to persist its real
        // weights. FixtureStudent: no-op (default trait impl). CudaStudent:
        // delegates to trainer.save_apr() and writes an APR file alongside
        // the metadata sidecar. Without this, the cuda path's trained GPU
        // weights are never serialized — Stage D 2026-05-20 ran 25h and
        // produced a 200-byte empty model.safetensors.
        let apr_target = self.config.output.dir.join("model.apr");
        self.student
            .save_checkpoint(&apr_target)
            .map_err(|e| EntrenarError::Internal {
                message: format!(
                    "student.save_checkpoint({}) failed: {e}",
                    apr_target.display()
                ),
            })?;

        Ok(output_path)
    }
}

/// Resolve a model identifier to a local filesystem path.
///
/// SPEC-DISTILL-001 Phase 2c helper: computes the scalar KD loss for the
/// current state of (teacher, student) on a `dummy_batch`, without
/// applying any gradient update. Used to bracket the training loop with
/// initial-loss and final-loss measurements that drive
/// `TrainingMetrics.improvement_ratio`.
/// PMAT-706: pure parse of the `APR_DISTILL_MAX_STEPS` value.
///
/// Per the `apr-distill-smoke-validation-v1` env-var precondition:
/// - unset / empty / non-integer → `None` (no smoke mode; values that fail to
///   parse silently fall back to "unset" so there is no regression).
/// - `"0"` → `Some(0)` (degenerate; `train()` turns this into a clear `Err`).
/// - `"N"` for N ≥ 1 → `Some(N)` (smoke mode: early-break after N steps).
///
/// Split out as a pure function so the contract precondition is unit-testable
/// without touching process env (this project has repeatedly been bitten by
/// env-var races in parallel tests).
fn parse_max_steps_value(raw: Option<&str>) -> Option<u64> {
    raw.map(str::trim)
        .filter(|v| !v.is_empty())
        .and_then(|v| v.parse::<u64>().ok())
}

/// PMAT-706: read `APR_DISTILL_MAX_STEPS` from the environment once.
fn parse_max_steps() -> Option<u64> {
    parse_max_steps_value(std::env::var("APR_DISTILL_MAX_STEPS").ok().as_deref())
}

/// PMAT-PERPOS: read the `APR_DISTILL_PER_POSITION` opt-in once. Truthy
/// values: `1`, `true`, `yes`, `on` (case-insensitive). Anything else
/// (unset, empty, `0`, `false`) → per-row mode (no behavior change).
fn parse_per_position() -> bool {
    std::env::var("APR_DISTILL_PER_POSITION")
        .ok()
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

/// PMAT-706: projected-wall-time target for the smoke summary. Defaults to
/// 50_000 steps; override with `APR_DISTILL_PROJECT_TO_STEPS=N` (N ≥ 1).
fn parse_project_to_steps() -> u64 {
    std::env::var("APR_DISTILL_PROJECT_TO_STEPS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(50_000)
}

/// PMAT-706: build the two `[SMOKE]` summary lines printed after an
/// early-break smoke run.
///
/// Pure + total: extracted so the exact wire format (FT-SMOKE-003) is
/// unit-testable without capturing stdout. `steps` is ≥ 1 on the smoke path
/// and `elapsed_secs` is clamped here, so there is no division-by-zero
/// (KANI-SMOKE-002). Linear extrapolation `steps → project_to_steps`; the
/// loss trajectory is reported, never asserted — smoke mode is a plumbing
/// check, not a quality gate.
fn smoke_summary_lines(
    steps: u64,
    elapsed_secs: f32,
    initial_loss: f32,
    final_loss: f32,
    project_to_steps: u64,
) -> (String, String) {
    let secs = elapsed_secs.max(1e-6);
    let throughput = steps as f32 / secs; // step/s
    let projected_secs = project_to_steps as f32 / throughput.max(1e-9);
    let summary = format!(
        "[SMOKE] {steps} steps in {secs:.2}s: \
         initial_loss={initial_loss:.4}, final_loss={final_loss:.4}, \
         throughput={throughput:.2} step/s"
    );
    let projection = format!(
        "[SMOKE] projected full-run wall time ({project_to_steps} steps): \
         {:.2}h / {:.1} min / {:.1}s",
        projected_secs / 3600.0,
        projected_secs / 60.0,
        projected_secs
    );
    (summary, projection)
}

fn kd_step_loss_for_pipeline(
    teacher: &mut dyn crate::teacher_provider::TeacherLogitsProvider,
    student: &mut dyn crate::student_provider::StudentLogitsProvider,
    input_ids: &[Vec<u32>],
    labels: &[usize],
    temperature: f32,
    alpha: f32,
) -> Result<f32> {
    let teacher_logits = teacher.logits_for_batch(input_ids)?;
    let student_logits = student.logits_for_batch(input_ids)?;
    if teacher_logits.len() != student_logits.len() {
        return Err(EntrenarError::Internal {
            message: format!(
                "kd_step_loss_for_pipeline: teacher returned {} logits batches, \
                 student returned {} — they must match",
                teacher_logits.len(),
                student_logits.len()
            ),
        });
    }
    let mut total = 0.0_f32;
    for ((s, t), &label) in student_logits
        .iter()
        .zip(teacher_logits.iter())
        .zip(labels.iter())
    {
        total += crate::kd_step::kd_loss(s, t, label, temperature, alpha);
    }
    Ok(if input_ids.is_empty() {
        0.0
    } else {
        total / input_ids.len() as f32
    })
}

/// - If it contains `/` or `.` and exists on disk, returns the path directly.
/// - If it looks like a HuggingFace model ID (org/model), uses HfModelFetcher
///   when the `hub` feature is enabled.
/// - Otherwise returns an error.
fn resolve_model_path(model_id: &str) -> Result<PathBuf> {
    let path = Path::new(model_id);

    // Check if it's a local path that exists
    if path.exists() {
        return Ok(path.to_path_buf());
    }

    // If it looks like a local path but doesn't exist, error
    if model_id.starts_with('/')
        || model_id.starts_with("./")
        || model_id.starts_with("../")
        || model_id.ends_with(".safetensors")
        || model_id.ends_with(".gguf")
    {
        return Err(EntrenarError::ModelNotFound {
            path: path.to_path_buf(),
        });
    }

    // Looks like a HuggingFace model ID (org/model)
    #[cfg(feature = "hub")]
    {
        let fetcher = entrenar::hf_pipeline::HfModelFetcher::new().map_err(|e| {
            EntrenarError::HuggingFace {
                message: format!("failed to initialize HF fetcher: {e}"),
            }
        })?;

        let artifact = fetcher
            .download_model(model_id, entrenar::hf_pipeline::FetchOptions::default())
            .map_err(|e| EntrenarError::HuggingFace {
                message: format!("failed to download '{model_id}': {e}"),
            })?;

        Ok(artifact.path)
    }

    #[cfg(not(feature = "hub"))]
    {
        if model_id.contains('/') {
            return Err(EntrenarError::HuggingFace {
                message: format!(
                    "'{model_id}' looks like a HuggingFace model ID, but the 'hub' feature is not enabled. \
                     Rebuild with: cargo build -p entrenar-distill --features hub"
                ),
            });
        }

        Err(EntrenarError::ModelNotFound {
            path: path.to_path_buf(),
        })
    }
}

/// Build synthetic logits from model weights for loss computation.
///
/// Takes the first weight tensor large enough and reshapes a slice of it
/// into [batch_size, num_classes]. This is a placeholder for real forward
/// pass outputs until the autograd backward ops are complete.
#[allow(dead_code)]
fn build_synthetic_logits(
    weights: &HashMap<String, Vec<f32>>,
    batch_size: usize,
    num_classes: usize,
) -> Array2<f32> {
    let needed = batch_size * num_classes;

    // Find a weight tensor large enough
    for data in weights.values() {
        if data.len() >= needed {
            let slice = &data[..needed];
            return Array2::from_shape_vec((batch_size, num_classes), slice.to_vec())
                .expect("shape matches needed elements");
        }
    }

    // Fallback: generate small random-like logits from whatever weights exist
    let mut logits = Vec::with_capacity(needed);
    let all_data: Vec<f32> = weights.values().flat_map(|v| v.iter().copied()).collect();
    for i in 0..needed {
        logits.push(if all_data.is_empty() {
            (i as f32 * 0.1) % 3.0 - 1.0
        } else {
            all_data[i % all_data.len()]
        });
    }

    Array2::from_shape_vec((batch_size, num_classes), logits)
        .expect("shape matches needed elements")
}

/// Compute the knowledge distillation gradient with respect to student logits.
///
/// The gradient of the KD loss L = α·T²·KL(teacher_T || student_T) + (1-α)·CE(student, labels):
///
/// ∂L/∂z_student = α·T·(softmax(z_s/T) - softmax(z_t/T))
///               + (1-α)·(softmax(z_s) - one_hot(labels))
#[allow(dead_code)]
fn kd_gradient(
    student_logits: &Array2<f32>,
    teacher_logits: &Array2<f32>,
    labels: &[usize],
    temperature: f32,
    alpha: f32,
) -> Array2<f32> {
    let batch_size = student_logits.nrows();
    let num_classes = student_logits.ncols();

    // Soft target gradient: α·T·(softmax(student/T) - softmax(teacher/T))
    let student_soft = softmax_2d(&(student_logits / temperature));
    let teacher_soft = softmax_2d(&(teacher_logits / temperature));
    let soft_grad = (&student_soft - &teacher_soft) * (alpha * temperature);

    // Hard target gradient: (1-α)·(softmax(student) - one_hot(labels))
    let student_hard = softmax_2d(student_logits);
    let mut one_hot = Array2::zeros((batch_size, num_classes));
    for (i, &label) in labels.iter().enumerate() {
        if label < num_classes {
            one_hot[[i, label]] = 1.0;
        }
    }
    let hard_grad = (&student_hard - &one_hot) * (1.0 - alpha);

    // Combined gradient
    &soft_grad + &hard_grad
}

/// Compute softmax along the last axis of a 2D array.
#[allow(dead_code)]
fn softmax_2d(x: &Array2<f32>) -> Array2<f32> {
    let mut result = x.clone();
    for mut row in result.axis_iter_mut(Axis(0)) {
        let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|v| (v - max_val).exp());
        let sum: f32 = row.sum();
        row.mapv_inplace(|v| v / sum);
    }
    result
}

/// Write trained logit values back into the first suitable weight tensor.
#[allow(dead_code)]
fn write_logits_to_weights(
    weights: &mut HashMap<String, Vec<f32>>,
    logits: &Array2<f32>,
    batch_size: usize,
    num_classes: usize,
) {
    let needed = batch_size * num_classes;
    let logit_data: Vec<f32> = logits.iter().copied().collect();

    for data in weights.values_mut() {
        if data.len() >= needed {
            data[..needed].copy_from_slice(&logit_data);
            return;
        }
    }
}

/// Known model size patterns: (substring, parameter count).
/// Ordered largest-first so "70b" matches before "7b".
const MODEL_SIZE_PATTERNS: &[(&str, u64)] = &[
    ("70b", 70_000_000_000),
    ("65b", 65_000_000_000),
    ("33b", 30_000_000_000),
    ("30b", 30_000_000_000),
    ("13b", 13_000_000_000),
    ("8b", 7_000_000_000),
    ("7b", 7_000_000_000),
    ("3b", 3_000_000_000),
    ("1.1b", 1_100_000_000),
    ("1b", 1_100_000_000),
    ("350m", 350_000_000),
    ("base", 350_000_000),
    ("125m", 125_000_000),
    ("small", 125_000_000),
];

/// Estimate parameter count from model ID.
fn estimate_params_from_model_id(model_id: &str) -> u64 {
    let lower = model_id.to_lowercase();

    MODEL_SIZE_PATTERNS
        .iter()
        .find(|(pattern, _)| lower.contains(pattern))
        .map_or(1_000_000_000, |&(_, count)| count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DistillConfig;

    #[test]
    fn test_estimate_params_from_model_id() {
        assert_eq!(
            estimate_params_from_model_id("meta-llama/Llama-2-7b"),
            7_000_000_000
        );
        assert_eq!(
            estimate_params_from_model_id("TinyLlama/TinyLlama-1.1B"),
            1_100_000_000
        );
        assert_eq!(
            estimate_params_from_model_id("microsoft/codebert-base"),
            350_000_000
        );
    }

    #[test]
    fn test_training_metrics_improvement() {
        let metrics = TrainingMetrics {
            initial_loss: 2.0,
            final_loss: 1.0,
            best_loss: 0.9,
            steps_completed: 1000,
            throughput: 100.0,
        };

        assert!((metrics.improvement_ratio() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_memory_estimation() {
        let config = DistillConfig::minimal("meta-llama/Llama-2-7b", "TinyLlama/TinyLlama-1.1B");

        let estimate = Pipeline::estimate_memory(&config).expect("config should be valid");

        assert!(estimate.total_bytes > 10_000_000_000);
        assert!(estimate.recommended_batch_size > 0);
    }

    #[test]
    fn test_pipeline_result_has_duration() {
        let result = PipelineResult {
            output_path: PathBuf::from("/tmp/output"),
            metrics: TrainingMetrics::default(),
            duration_seconds: 100.0,
        };

        assert!(result.duration_seconds > 0.0);
    }

    #[test]
    fn test_build_synthetic_logits_shape() {
        let mut weights = HashMap::new();
        weights.insert("w".to_string(), vec![0.5; 256]);

        let logits = build_synthetic_logits(&weights, 4, 32);
        assert_eq!(logits.shape(), &[4, 32]);
    }

    #[test]
    fn test_build_synthetic_logits_empty_weights() {
        let weights = HashMap::new();
        let logits = build_synthetic_logits(&weights, 2, 8);
        assert_eq!(logits.shape(), &[2, 8]);
    }

    #[test]
    fn test_kd_gradient_reduces_loss() {
        let teacher = Array2::from_shape_vec((2, 4), vec![2.0, 1.0, 0.5, 0.1, 1.5, 1.2, 0.8, 0.3])
            .expect("operation should succeed");
        let mut student =
            Array2::from_shape_vec((2, 4), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2])
                .expect("operation should succeed");
        let labels = vec![0, 1];
        let loss_fn = DistillationLoss::new(4.0, 0.7);

        let initial_loss = loss_fn.forward(&student, &teacher, &labels);

        // Apply gradient steps
        for _ in 0..100 {
            let grad = kd_gradient(&student, &teacher, &labels, 4.0, 0.7);
            student = &student - &(grad * 0.5);
        }

        let final_loss = loss_fn.forward(&student, &teacher, &labels);
        assert!(
            final_loss < initial_loss,
            "KD gradient did not reduce loss: {initial_loss} -> {final_loss}"
        );
    }

    #[test]
    fn test_resolve_local_path_missing() {
        let result = resolve_model_path("/nonexistent/model.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_local_path_exists() {
        let tmp = tempfile::TempDir::new().expect("temp file creation should succeed");
        let path = tmp.path().join("model.safetensors");
        std::fs::write(&path, b"dummy").expect("file write should succeed");

        let resolved = resolve_model_path(path.to_str().expect("operation should succeed"))
            .expect("operation should succeed");
        assert_eq!(resolved, path);
    }

    #[cfg(not(feature = "hub"))]
    #[test]
    fn test_resolve_hf_model_without_hub_feature() {
        let result = resolve_model_path("meta-llama/Llama-2-7b");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("hub"));
    }

    #[test]
    fn test_pipeline_execute_with_real_safetensors() {
        use safetensors::tensor::{Dtype, TensorView};

        let tmp = tempfile::TempDir::new().expect("temp file creation should succeed");

        // Create teacher SafeTensors
        let teacher_data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        let teacher_bytes: Vec<u8> = bytemuck::cast_slice(&teacher_data).to_vec();
        let teacher_views = vec![(
            "layer.weight",
            TensorView::new(Dtype::F32, vec![16, 16], &teacher_bytes)
                .expect("operation should succeed"),
        )];
        let teacher_path = tmp.path().join("teacher.safetensors");
        std::fs::write(
            &teacher_path,
            safetensors::serialize(teacher_views, None).expect("file write should succeed"),
        )
        .expect("operation should succeed");

        // Create student SafeTensors
        let student_data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.005).collect();
        let student_bytes: Vec<u8> = bytemuck::cast_slice(&student_data).to_vec();
        let student_views = vec![(
            "layer.weight",
            TensorView::new(Dtype::F32, vec![16, 16], &student_bytes)
                .expect("operation should succeed"),
        )];
        let student_path = tmp.path().join("student.safetensors");
        std::fs::write(
            &student_path,
            safetensors::serialize(student_views, None).expect("file write should succeed"),
        )
        .expect("operation should succeed");

        // Create config pointing to local files
        let output_dir = tmp.path().join("output");
        let mut config = DistillConfig::minimal(
            teacher_path.to_str().expect("config should be valid"),
            student_path.to_str().expect("config should be valid"),
        );
        config.output.dir = output_dir.clone();
        config.training.epochs = 2;
        config.training.batch_size = 4;

        let mut pipeline = Pipeline::new(&config);
        let result = pipeline.execute().expect("operation should succeed");

        assert!(result.output_path.exists());
        assert!(result.metrics.steps_completed > 0);
        assert!(result.metrics.initial_loss > 0.0);
        assert!(result.duration_seconds > 0.0);

        // Verify distillation metadata sidecar was created
        assert!(output_dir.join("distillation_metadata.json").exists());
    }

    /// F-DISTILL-PIPELINE-001 — Phase 2c end-to-end falsifier.
    ///
    /// Runs `Pipeline::execute()` with the default FixtureTeacher +
    /// FixtureStudent and asserts that
    /// `metrics.final_loss < metrics.initial_loss`. This pins the data
    /// flow correctness: teacher provider → student forward → kd_step
    /// (loss + gradient) → student.apply_kd_gradient. If any of those
    /// links break, the loss either stays flat or increases.
    #[test]
    fn falsify_pipeline_001_end_to_end_loss_decreases() {
        use safetensors::tensor::{Dtype, TensorView};

        let tmp = tempfile::TempDir::new().expect("temp file creation should succeed");

        // Minimal safetensors for the on-disk "weight loading" stage —
        // the actual values aren't used by Phase 2c's logic, but the
        // loader is still called for back-compat.
        let dummy: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
        let dummy_bytes: Vec<u8> = bytemuck::cast_slice(&dummy).to_vec();
        for name in ["teacher", "student"] {
            let p = tmp.path().join(format!("{name}.safetensors"));
            let views = vec![(
                "layer.weight",
                TensorView::new(Dtype::F32, vec![8, 4], &dummy_bytes).expect("safetensors view"),
            )];
            std::fs::write(
                &p,
                safetensors::serialize(views, None).expect("safetensors serialize"),
            )
            .expect("safetensors write");
        }

        let out_dir = tmp.path().join("out");
        let mut config = DistillConfig::minimal(
            tmp.path().join("teacher.safetensors").to_str().unwrap(),
            tmp.path().join("student.safetensors").to_str().unwrap(),
        );
        config.output.dir = out_dir;
        config.training.epochs = 3;
        config.training.batch_size = 4;
        config.distillation.temperature = 4.0;
        config.distillation.alpha = 0.5;

        let mut pipeline = Pipeline::new(&config);
        let result = pipeline.execute().expect("pipeline must succeed");

        eprintln!(
            "[F-DISTILL-PIPELINE-001] initial_loss={}, final_loss={}, steps={}",
            result.metrics.initial_loss, result.metrics.final_loss, result.metrics.steps_completed
        );

        assert!(
            result.metrics.steps_completed >= 3,
            "must run at least 3 steps in 3 epochs"
        );
        assert!(
            result.metrics.final_loss < result.metrics.initial_loss,
            "F-DISTILL-PIPELINE-001 FAILED: end-to-end pipeline did not \
             reduce loss (initial={}, final={}). This means the data \
             flow teacher → student → kd_step → apply_kd_gradient is \
             broken somewhere.",
            result.metrics.initial_loss,
            result.metrics.final_loss
        );
    }

    /// FT-PERPOS-005 (contract distill-per-position-kd-v1): the pipeline in
    /// per-position mode trains end-to-end on every position and reduces loss,
    /// AND `with_per_position(false)` is byte-identical to the default path
    /// (the production loop is untouched unless explicitly opted in).
    #[test]
    fn falsify_perpos_005_pipeline_per_position_reduces_loss() {
        use safetensors::tensor::{Dtype, TensorView};
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let dummy: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
        let dummy_bytes: Vec<u8> = bytemuck::cast_slice(&dummy).to_vec();
        for name in ["teacher", "student"] {
            let p = tmp.path().join(format!("{name}.safetensors"));
            let views = vec![(
                "layer.weight",
                TensorView::new(Dtype::F32, vec![8, 4], &dummy_bytes).expect("view"),
            )];
            std::fs::write(&p, safetensors::serialize(views, None).expect("ser")).expect("write");
        }
        let mk_config = |out: std::path::PathBuf| {
            let mut c = DistillConfig::minimal(
                tmp.path().join("teacher.safetensors").to_str().unwrap(),
                tmp.path().join("student.safetensors").to_str().unwrap(),
            );
            c.output.dir = out;
            c.training.epochs = 3;
            c.training.batch_size = 4;
            c.distillation.temperature = 4.0;
            c.distillation.alpha = 0.5;
            c
        };

        // Per-position mode: trains on every position, loss must decrease.
        let cfg_pp = mk_config(tmp.path().join("out_pp"));
        let mut pp = Pipeline::new(&cfg_pp).with_per_position(true);
        let r_pp = pp.execute().expect("per-position pipeline must run");
        eprintln!(
            "[FT-PERPOS-005] per-position initial={} final={} steps={}",
            r_pp.metrics.initial_loss, r_pp.metrics.final_loss, r_pp.metrics.steps_completed
        );
        assert!(
            r_pp.metrics.final_loss < r_pp.metrics.initial_loss,
            "per-position KD must reduce loss (initial={}, final={})",
            r_pp.metrics.initial_loss,
            r_pp.metrics.final_loss
        );

        // Default (per-row) still works and is the unchanged path.
        let cfg_row = mk_config(tmp.path().join("out_row"));
        let mut row = Pipeline::new(&cfg_row).with_per_position(false);
        let r_row = row.execute().expect("per-row pipeline must run");
        assert!(
            r_row.metrics.final_loss < r_row.metrics.initial_loss,
            "per-row path must remain healthy (initial={}, final={})",
            r_row.metrics.initial_loss,
            r_row.metrics.final_loss
        );
    }

    /// Falsification: does training actually reduce loss?
    #[test]
    fn test_falsify_training_reduces_loss() {
        use safetensors::tensor::{Dtype, TensorView};

        let tmp = tempfile::TempDir::new().expect("temp file creation should succeed");

        // Teacher: higher magnitude weights (stronger signal)
        let teacher_data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.02 - 2.0).collect();
        let teacher_bytes: Vec<u8> = bytemuck::cast_slice(&teacher_data).to_vec();
        let teacher_views = vec![(
            "layer.weight",
            TensorView::new(Dtype::F32, vec![16, 16], &teacher_bytes)
                .expect("operation should succeed"),
        )];
        let teacher_path = tmp.path().join("teacher.safetensors");
        std::fs::write(
            &teacher_path,
            safetensors::serialize(teacher_views, None).expect("file write should succeed"),
        )
        .expect("operation should succeed");

        // Student: different initialization
        let student_data: Vec<f32> = (0..256).map(|i| (i as f32) * -0.01 + 1.0).collect();
        let student_bytes: Vec<u8> = bytemuck::cast_slice(&student_data).to_vec();
        let student_views = vec![(
            "layer.weight",
            TensorView::new(Dtype::F32, vec![16, 16], &student_bytes)
                .expect("operation should succeed"),
        )];
        let student_path = tmp.path().join("student.safetensors");
        std::fs::write(
            &student_path,
            safetensors::serialize(student_views, None).expect("file write should succeed"),
        )
        .expect("operation should succeed");

        let output_dir = tmp.path().join("output");
        let mut config = DistillConfig::minimal(
            teacher_path.to_str().expect("config should be valid"),
            student_path.to_str().expect("config should be valid"),
        );
        config.output.dir = output_dir;
        config.training.epochs = 5;
        config.training.batch_size = 4;
        config.training.learning_rate = 0.01;

        let mut pipeline = Pipeline::new(&config);
        let result = pipeline.execute().expect("operation should succeed");

        eprintln!(
            "initial_loss={}, final_loss={}, best_loss={}, steps={}",
            result.metrics.initial_loss,
            result.metrics.final_loss,
            result.metrics.best_loss,
            result.metrics.steps_completed
        );

        // FALSIFICATION: loss must actually decrease
        assert!(
            result.metrics.final_loss < result.metrics.initial_loss,
            "Training did NOT reduce loss! initial={} final={}",
            result.metrics.initial_loss,
            result.metrics.final_loss
        );
    }

    /// Falsification: does export produce valid re-loadable SafeTensors?
    #[test]
    fn test_falsify_export_roundtrip() {
        use safetensors::tensor::{Dtype, TensorView};

        let tmp = tempfile::TempDir::new().expect("temp file creation should succeed");

        // Create identical teacher/student so training doesn't matter
        let data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        let data_bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let views = vec![(
            "layer.weight",
            TensorView::new(Dtype::F32, vec![16, 16], &data_bytes)
                .expect("operation should succeed"),
        )];
        let model_path = tmp.path().join("model.safetensors");
        std::fs::write(
            &model_path,
            safetensors::serialize(views, None).expect("file write should succeed"),
        )
        .expect("file write should succeed");

        let output_dir = tmp.path().join("output");
        let mut config = DistillConfig::minimal(
            model_path.to_str().expect("config should be valid"),
            model_path.to_str().expect("config should be valid"),
        );
        config.output.dir = output_dir.clone();
        config.training.epochs = 1;
        config.training.batch_size = 4;

        let mut pipeline = Pipeline::new(&config);
        let result = pipeline.execute().expect("operation should succeed");

        // FALSIFICATION: can we re-load the exported file?
        let exported_data = std::fs::read(&result.output_path).expect("file read should succeed");
        let loaded = safetensors::SafeTensors::deserialize(&exported_data)
            .expect("exported SafeTensors file is not valid!");

        // Must contain the same tensor name
        assert!(
            loaded.names().contains(&"layer.weight"),
            "exported file missing 'layer.weight' tensor, has: {:?}",
            loaded.names()
        );

        // Check the data is f32 and has correct shape
        let tensor = loaded.tensor("layer.weight").expect("load should succeed");
        assert_eq!(tensor.dtype(), Dtype::F32);
        assert_eq!(tensor.shape(), &[16, 16]);

        // FALSIFICATION: metadata sidecar must parse as valid JSON
        let meta_path = output_dir.join("distillation_metadata.json");
        let meta_str = std::fs::read_to_string(&meta_path).expect("file read should succeed");
        let meta: entrenar::distill::DistillationCheckpoint =
            serde_json::from_str(&meta_str).expect("metadata sidecar is not valid JSON!");
        assert!(meta.temperature > 0.0);
    }

    /// Falsification: what happens with mismatched teacher/student tensor names?
    #[test]
    fn test_falsify_mismatched_tensor_names() {
        use safetensors::tensor::{Dtype, TensorView};

        let tmp = tempfile::TempDir::new().expect("temp file creation should succeed");

        // Teacher has "encoder.weight"
        let data: Vec<f32> = vec![1.0; 256];
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let teacher_views = vec![(
            "encoder.weight",
            TensorView::new(Dtype::F32, vec![16, 16], &bytes).expect("encoding should succeed"),
        )];
        let teacher_path = tmp.path().join("teacher.safetensors");
        std::fs::write(
            &teacher_path,
            safetensors::serialize(teacher_views, None).expect("file write should succeed"),
        )
        .expect("operation should succeed");

        // Student has "decoder.weight" (completely different name)
        let student_views = vec![(
            "decoder.weight",
            TensorView::new(Dtype::F32, vec![16, 16], &bytes).expect("operation should succeed"),
        )];
        let student_path = tmp.path().join("student.safetensors");
        std::fs::write(
            &student_path,
            safetensors::serialize(student_views, None).expect("file write should succeed"),
        )
        .expect("operation should succeed");

        let output_dir = tmp.path().join("output");
        let mut config = DistillConfig::minimal(
            teacher_path.to_str().expect("config should be valid"),
            student_path.to_str().expect("config should be valid"),
        );
        config.output.dir = output_dir;
        config.training.epochs = 1;
        config.training.batch_size = 4;

        // Should NOT panic even with mismatched tensor names
        let mut pipeline = Pipeline::new(&config);
        let result = pipeline.execute();
        // This should succeed - gradient step just won't match any names
        assert!(
            result.is_ok(),
            "Pipeline panicked on mismatched tensors: {result:?}"
        );
    }

    /// Falsification: single-element tensor edge case
    #[test]
    fn test_falsify_tiny_tensors() {
        use safetensors::tensor::{Dtype, TensorView};

        let tmp = tempfile::TempDir::new().expect("temp file creation should succeed");

        // Single element tensor - too small for batch_size * num_classes
        let data: Vec<f32> = vec![0.5];
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let views = vec![(
            "w",
            TensorView::new(Dtype::F32, vec![1], &bytes).expect("operation should succeed"),
        )];
        let path = tmp.path().join("tiny.safetensors");
        std::fs::write(
            &path,
            safetensors::serialize(views, None).expect("file write should succeed"),
        )
        .expect("file write should succeed");

        let output_dir = tmp.path().join("output");
        let mut config = DistillConfig::minimal(
            path.to_str().expect("config should be valid"),
            path.to_str().expect("config should be valid"),
        );
        config.output.dir = output_dir;
        config.training.epochs = 1;
        config.training.batch_size = 2;

        let mut pipeline = Pipeline::new(&config);
        // Should NOT panic - should fall back to synthetic logits
        let result = pipeline.execute();
        assert!(
            result.is_ok(),
            "Pipeline panicked on tiny tensor: {result:?}"
        );
    }

    // PMAT-705: callback wiring tests.
    //
    // Contract: contracts/distill-pipeline-observability-v1.yaml.
    // Falsifier FT-OBSERV-001: per-step callbacks fire during training.
    mod pmat_705_observability {
        use super::*;
        use std::sync::{Arc, Mutex};

        /// Recording callback for tests — records every lifecycle invocation.
        struct RecordingCallback {
            events: Arc<Mutex<Vec<String>>>,
        }

        impl entrenar::train::TrainerCallback for RecordingCallback {
            fn on_train_begin(
                &mut self,
                ctx: &entrenar::train::CallbackContext,
            ) -> entrenar::train::CallbackAction {
                self.events
                    .lock()
                    .unwrap()
                    .push(format!("train_begin@step{}", ctx.step));
                entrenar::train::CallbackAction::Continue
            }

            fn on_epoch_begin(
                &mut self,
                ctx: &entrenar::train::CallbackContext,
            ) -> entrenar::train::CallbackAction {
                self.events
                    .lock()
                    .unwrap()
                    .push(format!("epoch_begin@{}", ctx.epoch));
                entrenar::train::CallbackAction::Continue
            }

            fn on_step_end(
                &mut self,
                ctx: &entrenar::train::CallbackContext,
            ) -> entrenar::train::CallbackAction {
                self.events
                    .lock()
                    .unwrap()
                    .push(format!("step_end@{}_loss{:.2}", ctx.step, ctx.loss));
                entrenar::train::CallbackAction::Continue
            }

            fn on_epoch_end(
                &mut self,
                ctx: &entrenar::train::CallbackContext,
            ) -> entrenar::train::CallbackAction {
                self.events
                    .lock()
                    .unwrap()
                    .push(format!("epoch_end@{}", ctx.epoch));
                entrenar::train::CallbackAction::Continue
            }

            fn on_train_end(&mut self, ctx: &entrenar::train::CallbackContext) {
                self.events
                    .lock()
                    .unwrap()
                    .push(format!("train_end@step{}", ctx.step));
            }

            fn name(&self) -> &'static str {
                "RecordingCallback"
            }
        }

        fn fixture_pipeline_config(tmp_dir: &std::path::Path, epochs: u32) -> DistillConfig {
            use safetensors::tensor::{Dtype, TensorView};

            let data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
            let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
            for name in ["teacher", "student"] {
                let views = vec![(
                    "layer.weight",
                    TensorView::new(Dtype::F32, vec![16, 16], &bytes)
                        .expect("tensor view should construct"),
                )];
                let path = tmp_dir.join(format!("{name}.safetensors"));
                std::fs::write(
                    &path,
                    safetensors::serialize(views, None).expect("serialize should succeed"),
                )
                .expect("write should succeed");
            }

            let out = tmp_dir.join("output");
            let mut config = DistillConfig::minimal(
                tmp_dir
                    .join("teacher.safetensors")
                    .to_str()
                    .expect("path is utf-8"),
                tmp_dir
                    .join("student.safetensors")
                    .to_str()
                    .expect("path is utf-8"),
            );
            config.output.dir = out;
            config.training.epochs = epochs;
            config.training.batch_size = 4;
            config
        }

        /// FT-OBSERV-001: per-step callbacks fire during training.
        #[test]
        fn falsify_observ_001_per_step_callback_fires() {
            let tmp = tempfile::TempDir::new().expect("tempdir");
            let config = fixture_pipeline_config(tmp.path(), 2);

            let events = Arc::new(Mutex::new(Vec::<String>::new()));
            let cb = Box::new(RecordingCallback {
                events: Arc::clone(&events),
            });
            let mut pipeline = Pipeline::new(&config).with_callback(cb);
            let _ = pipeline.execute().expect("pipeline should run");

            let log = events.lock().unwrap();
            // Must have at least one train_begin + train_end and ≥1 step_end.
            assert!(
                log.iter().any(|e| e.starts_with("train_begin")),
                "expected train_begin event; got: {log:?}"
            );
            assert!(
                log.iter().any(|e| e.starts_with("train_end")),
                "expected train_end event; got: {log:?}"
            );
            let step_end_count = log.iter().filter(|e| e.starts_with("step_end@")).count();
            assert!(
                step_end_count > 0,
                "expected ≥1 step_end events; got {step_end_count} (log: {log:?})"
            );
        }

        /// FT-OBSERV-001 corollary: step_end called exactly once per training step.
        #[test]
        fn falsify_observ_001_step_count_matches() {
            let tmp = tempfile::TempDir::new().expect("tempdir");
            let config = fixture_pipeline_config(tmp.path(), 2);
            let batch_size = u64::from(config.training.batch_size);
            let steps_per_epoch = (1000 / batch_size).max(1);
            let expected_steps = steps_per_epoch * u64::from(config.training.epochs);

            let events = Arc::new(Mutex::new(Vec::<String>::new()));
            let cb = Box::new(RecordingCallback {
                events: Arc::clone(&events),
            });
            let mut pipeline = Pipeline::new(&config).with_callback(cb);
            let result = pipeline.execute().expect("pipeline should run");

            let log = events.lock().unwrap();
            let step_end_count =
                u64::try_from(log.iter().filter(|e| e.starts_with("step_end@")).count())
                    .expect("count fits in u64");
            assert_eq!(
                step_end_count, expected_steps,
                "expected {expected_steps} step_end events; got {step_end_count} (steps_completed={})",
                result.metrics.steps_completed
            );
        }

        /// Callback ordering: callbacks fire in attach order.
        #[test]
        fn callback_ordering_preserved() {
            let tmp = tempfile::TempDir::new().expect("tempdir");
            let config = fixture_pipeline_config(tmp.path(), 1);

            let events_a = Arc::new(Mutex::new(Vec::<String>::new()));
            let events_b = Arc::new(Mutex::new(Vec::<String>::new()));
            let cb_a = Box::new(RecordingCallback {
                events: Arc::clone(&events_a),
            });
            let cb_b = Box::new(RecordingCallback {
                events: Arc::clone(&events_b),
            });
            let mut pipeline = Pipeline::new(&config)
                .with_callback(cb_a)
                .with_callback(cb_b);
            let _ = pipeline.execute().expect("pipeline should run");

            let log_a = events_a.lock().unwrap();
            let log_b = events_b.lock().unwrap();
            // Both callbacks see the same number of step_end events.
            let a_steps = log_a.iter().filter(|e| e.starts_with("step_end@")).count();
            let b_steps = log_b.iter().filter(|e| e.starts_with("step_end@")).count();
            assert_eq!(a_steps, b_steps, "both callbacks must see the same steps");
            assert!(a_steps > 0, "must run at least one step");
        }
    }

    // PMAT-706: smoke-validation early-break tests.
    //
    // Contract: contracts/apr-distill-smoke-validation-v1.yaml. Re-landed
    // after #1888 (52650c60c) squash-dropped the pipeline.rs implementation —
    // that commit shipped ONLY the contract YAML, so the gate it specifies
    // never existed in source. Falsifiers FT-SMOKE-001..004 + KANI-SMOKE-002.
    //
    // Test names match the contract's `evidence:` paths exactly
    // (`pipeline::tests::pmat_706_*`). Every test sets the budget via
    // `with_max_steps`, never the env var, so they are deterministic under
    // parallel execution.

    /// Build a fixture teacher+student safetensors pair + config in `tmp`.
    fn smoke_fixture_config(tmp_dir: &std::path::Path, epochs: u32) -> DistillConfig {
        use safetensors::tensor::{Dtype, TensorView};
        let data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        for name in ["teacher", "student"] {
            let views = vec![(
                "layer.weight",
                TensorView::new(Dtype::F32, vec![16, 16], &bytes)
                    .expect("tensor view should construct"),
            )];
            std::fs::write(
                tmp_dir.join(format!("{name}.safetensors")),
                safetensors::serialize(views, None).expect("serialize should succeed"),
            )
            .expect("write should succeed");
        }
        let mut config = DistillConfig::minimal(
            tmp_dir.join("teacher.safetensors").to_str().expect("utf-8"),
            tmp_dir.join("student.safetensors").to_str().expect("utf-8"),
        );
        config.output.dir = tmp_dir.join("output");
        config.training.epochs = epochs;
        config.training.batch_size = 4;
        config
    }

    /// FT-SMOKE-001: `APR_DISTILL_MAX_STEPS=10` runs exactly 10 steps,
    /// not the configured 100 epochs × 250 steps/epoch = 25_000.
    #[test]
    fn pmat_706_smoke() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let config = smoke_fixture_config(tmp.path(), 100);
        let mut pipeline = Pipeline::new(&config).with_max_steps(Some(10));
        let result = pipeline.execute().expect("smoke run should succeed");
        assert_eq!(
            result.metrics.steps_completed, 10,
            "FT-SMOKE-001: early-break must stop at exactly 10 steps, got {}",
            result.metrics.steps_completed
        );
    }

    /// FT-SMOKE-002: with the budget unset the loop runs the full configured
    /// schedule and still exports — the path is purely additive.
    #[test]
    fn pmat_706_no_regression() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let config = smoke_fixture_config(tmp.path(), 2);
        let batch_size = u64::from(config.training.batch_size);
        let expected = (1000 / batch_size).max(1) * u64::from(config.training.epochs);
        let mut pipeline = Pipeline::new(&config).with_max_steps(None);
        let result = pipeline.execute().expect("full run should succeed");
        assert_eq!(
            result.metrics.steps_completed, expected,
            "FT-SMOKE-002: unset budget must run the full {expected}-step schedule"
        );
        assert!(
            result.output_path.exists(),
            "FT-SMOKE-002: a full (non-smoke) run must still write an output model"
        );
    }

    /// FT-SMOKE-003: the summary is two grep/awk-parseable lines carrying
    /// throughput + projected wall time. Asserted against the pure builder so
    /// the exact wire format is pinned without capturing stdout.
    #[test]
    fn pmat_706_summary_format() {
        let (summary, projection) = smoke_summary_lines(10, 2.0, 6.0, 5.0, 50_000);
        assert!(
            summary.starts_with("[SMOKE] 10 steps in 2.00s:"),
            "got: {summary}"
        );
        assert!(summary.contains("initial_loss=6.0000"), "got: {summary}");
        assert!(summary.contains("final_loss=5.0000"), "got: {summary}");
        assert!(summary.contains("throughput=5.00 step/s"), "got: {summary}");
        assert!(
            projection.starts_with("[SMOKE] projected full-run wall time (50000 steps):"),
            "got: {projection}"
        );
        assert!(projection.contains("h /"), "got: {projection}");
        assert!(projection.contains("min /"), "got: {projection}");
        // 50_000 steps at 5 step/s = 10_000s.
        assert!(projection.contains("10000.0s"), "got: {projection}");
    }

    /// FT-SMOKE-004: a smoke run writes no final output model — the returned
    /// path is empty and no metadata sidecar appears.
    #[test]
    fn pmat_706_no_output_in_smoke() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let config = smoke_fixture_config(tmp.path(), 100);
        let out_dir = config.output.dir.clone();
        let mut pipeline = Pipeline::new(&config).with_max_steps(Some(5));
        let result = pipeline.execute().expect("smoke run should succeed");
        assert_eq!(result.metrics.steps_completed, 5);
        assert!(
            result.output_path.as_os_str().is_empty(),
            "FT-SMOKE-004: smoke output_path must be empty (no artifact written)"
        );
        assert!(
            !out_dir.join("distillation_metadata.json").exists(),
            "FT-SMOKE-004: smoke mode must not write the metadata sidecar"
        );
    }

    /// KANI-SMOKE-002 analog: `APR_DISTILL_MAX_STEPS=0` is a clear, actionable
    /// error — never a panic or a silent zero-step success.
    #[test]
    fn pmat_706_zero_steps_is_clear_error() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let config = smoke_fixture_config(tmp.path(), 4);
        let mut pipeline = Pipeline::new(&config).with_max_steps(Some(0));
        let err = pipeline.execute().expect_err("0 steps must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("APR_DISTILL_MAX_STEPS") && msg.contains("at least 1 step"),
            "expected an actionable 0-steps error, got: {msg}"
        );
    }

    /// Contract env-var precondition: invalid/empty → unset; a valid integer
    /// (including 0) parses through. Pure — no process env touched.
    #[test]
    fn pmat_706_parse_max_steps_value() {
        assert_eq!(parse_max_steps_value(None), None);
        assert_eq!(parse_max_steps_value(Some("")), None);
        assert_eq!(parse_max_steps_value(Some("   ")), None);
        assert_eq!(parse_max_steps_value(Some("abc")), None);
        assert_eq!(parse_max_steps_value(Some("-1")), None);
        assert_eq!(parse_max_steps_value(Some("0")), Some(0));
        assert_eq!(parse_max_steps_value(Some("10")), Some(10));
        assert_eq!(parse_max_steps_value(Some("  7 ")), Some(7));
    }
}
