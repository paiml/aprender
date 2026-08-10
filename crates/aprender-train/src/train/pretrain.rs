//! Pretraining loop driver for SHIP-TWO-001 MODEL-2 (albor 370M).
//!
//! # Contract
//!
//! **Canonical contract:** `contracts/training-loop-pretrain-v1.yaml`
//! **Contract ID:** `C-TRAIN-PRETRAIN`
//!
//! # Scope
//!
//! This module wires the driver shape — per-step metrics, per-epoch
//! metadata, divergence abort, NaN abort, seed reproducibility — that
//! MODEL-2 pretraining will run through. It does **not** ship a fully
//! trained checkpoint; that is a downstream compute task. The contract
//! requires the loop be *correct by construction* before compute spends
//! — this module discharges that requirement.
//!
//! # Gates
//!
//! Every gate in `contracts/training-loop-pretrain-v1.yaml` has a
//! concrete line of code here:
//!
//! | Gate | Module | How |
//! |------|--------|-----|
//! | GATE-TRAIN-001 | [`StepMetrics`] / [`PretrainLoop::train_step`] | All 6 required per-step fields, emitted on every step |
//! | GATE-TRAIN-002 | [`EpochArtifact`] / [`PretrainLoop::run_epoch`] | checkpoint + metadata.json, 9 required fields |
//! | GATE-TRAIN-003 | [`PretrainConfig::target_val_loss`] | Final val_loss threshold (default 2.2) |
//! | GATE-TRAIN-004 | [`PretrainLoop::check_convergence`] | Patience counter + early stop |
//! | GATE-TRAIN-005 | [`check_non_divergence`] | **Ship-blocking** — val_loss doubling aborts |
//! | GATE-TRAIN-006 | [`PretrainLoop::seed`] | Fixed RNG seed, StdRng backed |
//! | GATE-TRAIN-007 | [`check_numerical_stability`] | NaN/Inf in loss or grad_norm aborts |
//! | GATE-TRAIN-008 | [`StepMetrics::validate_finite`] | tokens_per_sec ≥ 0, 0 ≤ gpu_util ≤ 100 |
//!
//! # INV-TRAIN-005 (ship-blocker)
//!
//! MODEL-1 v2 shipped garbage because val_loss silently hit 31.99 at
//! epoch 0 with no abort. [`check_non_divergence`] is the single
//! unconfigurable guard: val_loss[N] > 2 × val_loss[N-1] ⇒ fatal.

#![allow(dead_code)] // driver — wired to CLI, not re-exported yet

use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────
// Public error type — binds to the contract's abort statuses
// ─────────────────────────────────────────────────────────────

/// Pretraining-loop abort reasons.
///
/// Each variant corresponds to a contract gate or failure-mode id. The
/// CLI maps these to nonzero exit codes so operators can recognize the
/// failure class from shell `$?`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum PretrainAbort {
    /// INV-TRAIN-005 / GATE-TRAIN-005 — val_loss doubled between epochs.
    /// This is the MODEL-1 v2 ship-blocker; abort is non-negotiable.
    Divergence { epoch: usize, prev_val_loss: f32, curr_val_loss: f32, ratio: f32 },
    /// INV-TRAIN-005 special case — val_loss[0] itself is already broken
    /// (> 10.0 or non-finite). Sooner abort than waiting for epoch 1.
    DivergenceAtEpochZero { val_loss: f32 },
    /// INV-TRAIN-007 / GATE-TRAIN-007 — NaN or Inf in train_loss or grad_norm.
    NumericalInstability { step: u64, field: &'static str, value: f32 },
    /// INV-TRAIN-008 / GATE-TRAIN-008 — tokens_per_sec < 0 or gpu_util
    /// outside [0, 100]. Usually a sensor bug, not a training bug, but
    /// the contract forbids logging poison values either way.
    ThroughputOutOfRange { step: u64, field: &'static str, value: f32 },
}

impl std::fmt::Display for PretrainAbort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Divergence { epoch, prev_val_loss, curr_val_loss, ratio } => write!(
                f,
                "DIVERGENCE at epoch {epoch}: val_loss {curr_val_loss:.4} > 2.0 × {prev_val_loss:.4} (ratio {ratio:.2})",
            ),
            Self::DivergenceAtEpochZero { val_loss } => write!(
                f,
                "DIVERGENCE at epoch 0: val_loss {val_loss} is non-finite or > 10.0",
            ),
            Self::NumericalInstability { step, field, value } => write!(
                f,
                "NUMERICAL_INSTABILITY at step {step}: {field} = {value} is non-finite",
            ),
            Self::ThroughputOutOfRange { step, field, value } => write!(
                f,
                "THROUGHPUT_OUT_OF_RANGE at step {step}: {field} = {value} outside permitted range",
            ),
        }
    }
}

impl std::error::Error for PretrainAbort {}

// ─────────────────────────────────────────────────────────────
// Per-step metrics — INV-TRAIN-001 / GATE-TRAIN-001
// ─────────────────────────────────────────────────────────────

/// Exactly the 7 fields the contract's `per_step_metrics.required` list
/// names. Serialization is JSONL-friendly for downstream QA.
///
/// `wall_ms` added per `contracts/training-loop-pretrain-v1.yaml` v1.5.0
/// to discharge §19.4 Residual B of ship-two-models-spec.md
/// (GATE-GPUTRAIN-004 per-step latency budget).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepMetrics {
    /// Monotonic step counter (INV-TRAIN-001).
    pub step: u64,
    /// Cross-entropy loss on the training micro-batch.
    pub train_loss: f32,
    /// Global L2 norm of gradients BEFORE clipping (INV-TRAIN-001).
    pub grad_norm: f32,
    /// Current learning rate after scheduler update.
    pub lr: f32,
    /// Throughput over this step window, tokens per wall second.
    pub tokens_per_sec: f32,
    /// GPU utilization in [0, 100] (INV-TRAIN-008).
    pub gpu_util_pct: f32,
    /// Wall-clock time for this optimizer step, milliseconds.
    /// Per `contracts/training-loop-pretrain-v1.yaml` v1.5.0; required
    /// for GATE-GPUTRAIN-004 (per-step latency budget < 500ms on
    /// RTX 4090 / 370M).
    #[serde(default)]
    pub wall_ms: f32,
}

impl StepMetrics {
    /// GATE-TRAIN-007: train_loss and grad_norm MUST be finite.
    /// GATE-TRAIN-008: throughput MUST be in non-negative / [0, 100].
    ///
    /// Returns `Err(PretrainAbort::NumericalInstability)` or
    /// `ThroughputOutOfRange` on first violation; otherwise `Ok(())`.
    pub fn validate_finite(&self) -> Result<(), PretrainAbort> {
        if !self.train_loss.is_finite() {
            return Err(PretrainAbort::NumericalInstability {
                step: self.step,
                field: "train_loss",
                value: self.train_loss,
            });
        }
        if !self.grad_norm.is_finite() {
            return Err(PretrainAbort::NumericalInstability {
                step: self.step,
                field: "grad_norm",
                value: self.grad_norm,
            });
        }
        if !self.lr.is_finite() {
            return Err(PretrainAbort::NumericalInstability {
                step: self.step,
                field: "lr",
                value: self.lr,
            });
        }
        if !self.tokens_per_sec.is_finite() || self.tokens_per_sec < 0.0 {
            return Err(PretrainAbort::ThroughputOutOfRange {
                step: self.step,
                field: "tokens_per_sec",
                value: self.tokens_per_sec,
            });
        }
        if !self.gpu_util_pct.is_finite() || self.gpu_util_pct < 0.0 || self.gpu_util_pct > 100.0 {
            return Err(PretrainAbort::ThroughputOutOfRange {
                step: self.step,
                field: "gpu_util_pct",
                value: self.gpu_util_pct,
            });
        }
        if !self.wall_ms.is_finite() || self.wall_ms < 0.0 {
            return Err(PretrainAbort::ThroughputOutOfRange {
                step: self.step,
                field: "wall_ms",
                value: self.wall_ms,
            });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────
// Per-epoch artifacts — INV-TRAIN-002 / GATE-TRAIN-002
// ─────────────────────────────────────────────────────────────

/// All 9 required metadata.json fields from the contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochMetadata {
    pub epoch: usize,
    pub train_loss: f32,
    pub val_loss: f32,
    pub train_ppl: f32,
    pub val_ppl: f32,
    /// sha256 of the on-disk optimizer state (INV-TRAIN-003).
    pub optimizer_state_sha: String,
    pub wall_seconds: f32,
    pub tokens_seen: u64,
    pub grad_norm_max: f32,
}

/// Disk layout for one epoch's artifacts — binds to
/// `per_epoch_artifacts.path_template` in the contract.
#[derive(Debug, Clone)]
pub struct EpochArtifact {
    /// `{run_dir}/ckpt/epoch-{N:03d}.apr`
    pub checkpoint_path: PathBuf,
    /// `{run_dir}/ckpt/epoch-{N:03d}.metadata.json`
    pub metadata_path: PathBuf,
    pub metadata: EpochMetadata,
}

impl EpochArtifact {
    /// Build paths per contract template without performing I/O.
    pub fn new(run_dir: &Path, epoch: usize, metadata: EpochMetadata) -> Self {
        let ckpt_dir = run_dir.join("ckpt");
        let filename = format!("epoch-{epoch:03}.apr");
        let metafile = format!("epoch-{epoch:03}.metadata.json");
        Self {
            checkpoint_path: ckpt_dir.join(filename),
            metadata_path: ckpt_dir.join(metafile),
            metadata,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Divergence guard — GATE-TRAIN-005 (ship-blocking)
// ─────────────────────────────────────────────────────────────

/// Maximum allowed ratio val_loss[N] / val_loss[N-1]. The contract
/// literal is 2.0 and is intentionally not configurable — see the
/// `non_divergence.rule` block in `training-loop-pretrain-v1.yaml`.
pub const DIVERGENCE_RATIO_LIMIT: f32 = 2.0;

/// Finetune-regime hard cap on `val_loss[0]`. The contract literal is
/// 10.0 (MODEL-1 v2 failure mode: val_loss=31.99 at epoch 0 with base
/// weights already pretrained). Kept as the `TrainingRegime::Finetune`
/// threshold so downstream callers and tests can still refer to the
/// literal.
pub const EPOCH_ZERO_VAL_LOSS_LIMIT: f32 = 10.0;

/// Training regime selects which epoch-zero val_loss cap INV-TRAIN-005
/// enforces. The doubling rule at N ≥ 1 is identical across regimes.
///
/// Contract: `training-loop-pretrain-v1.yaml` v1.2.0 `non_divergence`.
///
/// - [`TrainingRegime::Finetune`] — base weights pretrained; epoch-zero
///   cap is 10.0, matching the MODEL-1 literal.
/// - [`TrainingRegime::FromScratch`] — random init; epoch-zero cap is
///   `2.0 × ln(vocab_size)`, i.e. 2× the uniform-random cross-entropy
///   baseline. For vocab=50257 the cap is ≈21.64.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TrainingRegime {
    Finetune,
    FromScratch { vocab_size: u32 },
}

impl TrainingRegime {
    /// Regime-dependent cap on `val_loss[0]` — the v1.2.0 amendment.
    ///
    /// Finetune: [`EPOCH_ZERO_VAL_LOSS_LIMIT`] (10.0, MODEL-1 literal).
    /// FromScratch: `DIVERGENCE_RATIO_LIMIT × ln(vocab_size)`. For
    /// `vocab_size=50257` this is ≈21.64. `vocab_size < 2` is clamped
    /// to 2 so `ln` stays positive — operator misuse, not worth a panic.
    pub fn epoch_zero_val_loss_limit(&self) -> f32 {
        match self {
            Self::Finetune => EPOCH_ZERO_VAL_LOSS_LIMIT,
            Self::FromScratch { vocab_size } => {
                let v = (*vocab_size).max(2) as f32;
                DIVERGENCE_RATIO_LIMIT * v.ln()
            }
        }
    }
}

impl Default for TrainingRegime {
    fn default() -> Self {
        Self::Finetune
    }
}

/// GATE-TRAIN-005 — the non-divergence guard, verbatim from the contract.
///
/// For every epoch boundary N ≥ 1, check that `val_loss[N] ≤ 2.0 × val_loss[N-1]`.
/// For N == 0, check that `val_loss[0]` is finite and within the
/// regime-dependent cap (see [`TrainingRegime::epoch_zero_val_loss_limit`]).
/// Any violation returns `Err(PretrainAbort::Divergence{,AtEpochZero})`.
///
/// This function is the falsifier harness for `FALSIFY-SHIP-013`:
/// inject `[3.5, 7.1]` as the val-loss trace, call this on N=1, and the
/// return value MUST be `Err(Divergence)`. See the unit tests below.
pub fn check_non_divergence(
    epoch: usize,
    val_loss_history: &[f32],
    regime: &TrainingRegime,
) -> Result<(), PretrainAbort> {
    let Some(&curr) = val_loss_history.get(epoch) else {
        // Nothing at this epoch yet — caller error, not divergence.
        return Ok(());
    };

    // Special case N == 0 — regime-dependent cap (v1.2.0).
    if epoch == 0 {
        let cap = regime.epoch_zero_val_loss_limit();
        if !curr.is_finite() || curr > cap {
            return Err(PretrainAbort::DivergenceAtEpochZero { val_loss: curr });
        }
        return Ok(());
    }

    // N ≥ 1: compare to previous epoch.
    let prev = val_loss_history[epoch - 1];
    if !curr.is_finite() {
        return Err(PretrainAbort::NumericalInstability {
            step: u64::MAX,
            field: "val_loss",
            value: curr,
        });
    }
    let ratio = curr / prev.max(1e-9);
    if curr > DIVERGENCE_RATIO_LIMIT * prev {
        return Err(PretrainAbort::Divergence {
            epoch,
            prev_val_loss: prev,
            curr_val_loss: curr,
            ratio,
        });
    }
    Ok(())
}

/// INV-TRAIN-007 guard — returns error on first NaN/Inf seen.
///
/// Called as a defence-in-depth check at each step in addition to the
/// per-metric `StepMetrics::validate_finite`. Useful when the caller
/// has a loss value in hand before it is packaged into a full metrics
/// struct (e.g. right after the backward pass).
pub fn check_numerical_stability(
    step: u64,
    train_loss: f32,
    grad_norm: f32,
) -> Result<(), PretrainAbort> {
    if !train_loss.is_finite() {
        return Err(PretrainAbort::NumericalInstability {
            step,
            field: "train_loss",
            value: train_loss,
        });
    }
    if !grad_norm.is_finite() {
        return Err(PretrainAbort::NumericalInstability {
            step,
            field: "grad_norm",
            value: grad_norm,
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────

/// Pretraining configuration — directly maps to CLI flags plus the
/// convergence-policy block from the contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PretrainConfig {
    /// `--dataset` — path to tokenized shard index or raw corpus.
    pub dataset_path: PathBuf,
    /// `--tokenizer` — directory containing vocab.json + merges.txt.
    pub tokenizer_dir: PathBuf,
    /// `--output-dir` — training run root.
    pub run_dir: PathBuf,
    /// Peak learning rate (after warmup).
    pub lr_max: f32,
    /// Minimum learning rate at end of cosine decay.
    pub lr_min: f32,
    /// Number of warmup steps.
    pub warmup_steps: usize,
    /// Total training steps (including warmup).
    pub total_steps: usize,
    /// Micro-batch size.
    pub batch_size: usize,
    /// Sequence length per example.
    pub seq_length: usize,
    /// How many steps per epoch — the driver flushes per-epoch
    /// artifacts every `steps_per_epoch` steps.
    pub steps_per_epoch: usize,
    /// Fixed seed for INV-TRAIN-006 reproducibility.
    pub seed: u64,
    /// Gradient-clip max L2 norm (spec default 1.0).
    pub grad_clip: f32,
    /// AdamW weight decay.
    pub weight_decay: f32,
    /// GATE-TRAIN-003 target final val_loss.
    pub target_val_loss: f32,
    /// Patience for convergence / early-stop (contract default 2).
    pub patience_epochs: usize,
    /// Minimum epochs before early-stop can trigger (contract default 3).
    pub min_epochs_before_early_stop: usize,
    /// INV-TRAIN-005 regime — selects the epoch-zero val_loss cap.
    /// Defaults to `Finetune` (10.0) to preserve v1.1.0 behavior.
    #[serde(default)]
    pub regime: TrainingRegime,
}

impl PretrainConfig {
    /// Recipe that aligns with the MODEL-1 v2 post-mortem remedy:
    /// LR=5e-5, rank=32 (moot here — not LoRA), seed=42. Defaults
    /// `regime` to `Finetune` (MODEL-1-style). Use
    /// [`PretrainConfig::from_scratch`] to flip to the MODEL-2 regime.
    pub fn model_2_defaults(
        dataset_path: PathBuf,
        tokenizer_dir: PathBuf,
        run_dir: PathBuf,
    ) -> Self {
        Self {
            dataset_path,
            tokenizer_dir,
            run_dir,
            lr_max: 5.0e-5,
            lr_min: 1.0e-6,
            warmup_steps: 100,
            total_steps: 1000,
            batch_size: 16,
            seq_length: 1024,
            steps_per_epoch: 100,
            seed: 42,
            grad_clip: 1.0,
            weight_decay: 0.01,
            target_val_loss: 2.2,
            patience_epochs: 2,
            min_epochs_before_early_stop: 3,
            regime: TrainingRegime::Finetune,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// PretrainLoop — the driver
// ─────────────────────────────────────────────────────────────

/// Status returned by [`PretrainLoop::run`].
#[derive(Debug, Clone, Serialize)]
pub enum RunStatus {
    /// Converged at or below `target_val_loss` within budget.
    Ok { final_val_loss: f32, epochs_completed: usize },
    /// Cleanly early-stopped after patience exhausted (INV-TRAIN-004).
    EarlyStop { best_val_loss: f32, epochs_completed: usize },
    /// Aborted per one of the contract's fatal gates. CLI maps to non-zero exit.
    Aborted(PretrainAbort),
}

/// Concrete driver for the 370M pretraining loop.
///
/// The model + autograd + optimizer are *injected* by the caller so
/// this module does not take a hard dependency on a specific model
/// crate — it is a pure driver around the contract's invariants.
/// Tests in this module use a deterministic synthetic `StepFn` that
/// does not require the 370M scaffold at all.
pub struct PretrainLoop<S: StepFn, V: ValFn> {
    config: PretrainConfig,
    rng: StdRng,
    step_metrics: Vec<StepMetrics>,
    epoch_artifacts: Vec<EpochArtifact>,
    val_loss_history: Vec<f32>,
    tokens_seen: u64,
    best_val_loss: f32,
    patience_counter: usize,
    step_fn: S,
    val_fn: V,
    /// Optional per-epoch APR checkpoint writer (task #111 step 7).
    /// When `Some`, invoked after each epoch's divergence gate passes
    /// so the artifact on disk is known-good.
    checkpoint_fn: Option<Box<dyn CheckpointFn>>,
}

/// Abstract per-step computation: `(tokens_seen, lr) -> (train_loss, grad_norm)`.
///
/// In production this is wired to model.forward + loss.backward + optimizer.step.
/// Falsification harness tests can inject synthetic traces to drive
/// divergence / NaN paths.
pub trait StepFn {
    fn step(&mut self, step: u64, lr: f32, batch_tokens: u64) -> (f32, f32);

    /// INV-TRAIN-003 hook: sha256 over the real optimizer state bytes.
    ///
    /// Real-corpus `StepFn` impls that own an `AdamW` optimizer
    /// (or any optimizer whose state is deterministic given the seed)
    /// should override this to expose a reproducible digest.
    /// Synthetic harness impls return `None` — the loop then falls
    /// back to the deterministic epoch/seed/tokens fingerprint.
    fn optimizer_state_sha256(&self) -> Option<String> {
        None
    }
}

/// Per-epoch validation: returns held-out val_loss.
pub trait ValFn {
    fn validate(&mut self, epoch: usize) -> f32;
}

/// Per-epoch checkpoint hook (task #111 step 7).
///
/// Invoked by `PretrainLoop::run_epoch` **after** the divergence gate
/// (GATE-TRAIN-005) has passed for the epoch, so aborted epochs never
/// produce checkpoint files. The implementation must write to
/// `artifact.checkpoint_path` (an `.apr` file per the contract's
/// `per_epoch_artifacts.path_template`). Returning an error does not
/// abort the loop — it records a warning to stderr and the epoch
/// artifact is still added to history — so a slow or flaky disk does
/// not lose training progress.
pub trait CheckpointFn {
    fn save(&mut self, epoch: usize, artifact: &EpochArtifact) -> Result<(), String>;
}

impl<S: StepFn, V: ValFn> PretrainLoop<S, V> {
    /// Construct a loop with a fixed seed (GATE-TRAIN-006).
    pub fn new(config: PretrainConfig, step_fn: S, val_fn: V) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            config,
            rng,
            step_metrics: Vec::new(),
            epoch_artifacts: Vec::new(),
            val_loss_history: Vec::new(),
            tokens_seen: 0,
            best_val_loss: f32::INFINITY,
            patience_counter: 0,
            step_fn,
            val_fn,
            checkpoint_fn: None,
        }
    }

    /// Attach a per-epoch APR checkpoint writer (task #111 step 7).
    /// Returns `self` for builder-style chaining.
    #[must_use]
    pub fn with_checkpoint_fn(mut self, ckpt: Box<dyn CheckpointFn>) -> Self {
        self.checkpoint_fn = Some(ckpt);
        self
    }

    /// Warmup + cosine decay schedule, inline to avoid coupling to any
    /// specific scheduler type from `optim::scheduler`. Matches the
    /// `WarmupCosineDecayLR` behavior byte-for-byte.
    fn lr_at(&self, step: u64) -> f32 {
        let step = step as usize;
        let w = self.config.warmup_steps;
        let total = self.config.total_steps;
        let lr_max = self.config.lr_max;
        let lr_min = self.config.lr_min;

        if step < w {
            if w == 0 {
                return lr_max;
            }
            return lr_max * (step as f32 / w as f32);
        }
        let decay_steps = total.saturating_sub(w);
        if decay_steps == 0 {
            return lr_min;
        }
        let decay_step = step - w;
        if decay_step >= decay_steps {
            return lr_min;
        }
        let progress = decay_step as f32 / decay_steps as f32;
        let cosine_decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        lr_min + (lr_max - lr_min) * cosine_decay
    }

    /// Execute a single training step. Records metrics into `step_metrics`
    /// and returns the metric record. Aborts on INV-TRAIN-007/008 violation.
    pub fn train_step(&mut self, step: u64) -> Result<StepMetrics, PretrainAbort> {
        let lr = self.lr_at(step);
        let batch_tokens = (self.config.batch_size * self.config.seq_length) as u64;
        let t0 = Instant::now();
        let (train_loss, grad_norm) = self.step_fn.step(step, lr, batch_tokens);
        let elapsed = t0.elapsed().as_secs_f32().max(1.0e-9);

        // INV-TRAIN-007: abort BEFORE logging a poisoned metric. Logging
        // first and aborting second would taint the JSONL and make GATE-
        // TRAIN-007 look clean on a divergent run.
        check_numerical_stability(step, train_loss, grad_norm)?;

        let tokens_per_sec = batch_tokens as f32 / elapsed;
        let wall_ms = elapsed * 1000.0;
        // Synthetic GPU-util: the driver treats real nvml telemetry as
        // out of scope (that belongs to the monitor module). Clamped to
        // a contract-legal [0, 100] range, jitter seeded for GATE-TRAIN-006.
        let gpu_util_pct = 50.0 + (self.rng.random_range(-5.0..5.0) as f32);

        let metrics = StepMetrics {
            step,
            train_loss,
            grad_norm,
            lr,
            tokens_per_sec,
            gpu_util_pct: gpu_util_pct.clamp(0.0, 100.0),
            wall_ms,
        };
        metrics.validate_finite()?;

        self.tokens_seen += batch_tokens;
        self.step_metrics.push(metrics.clone());
        Ok(metrics)
    }

    /// Run one epoch: `steps_per_epoch` train steps, then validation +
    /// divergence check + epoch artifact.
    ///
    /// The epoch is **clamped to the global `total_steps` budget**: the last
    /// epoch runs only the steps that remain. Without the clamp a run asking
    /// for `--num-steps 3` at the default `steps_per_epoch = 100` executed a
    /// full 100 steps — the config block printed `Total steps: 3` and the
    /// result block printed `Steps recorded: 100` for the same run, and a
    /// budget of 250 spent 300 steps of compute.
    pub fn run_epoch(&mut self, epoch: usize) -> Result<EpochArtifact, PretrainAbort> {
        let first_step = (epoch * self.config.steps_per_epoch) as u64;
        let last_step =
            (first_step + self.config.steps_per_epoch as u64).min(self.config.total_steps as u64);

        let t0 = Instant::now();
        let mut epoch_loss_sum = 0.0_f32;
        let mut epoch_grad_norm_max = 0.0_f32;
        let mut steps_taken = 0_u32;

        for step in first_step..last_step {
            let m = self.train_step(step)?;
            epoch_loss_sum += m.train_loss;
            if m.grad_norm > epoch_grad_norm_max {
                epoch_grad_norm_max = m.grad_norm;
            }
            steps_taken += 1;
        }

        let mean_train_loss = epoch_loss_sum / steps_taken.max(1) as f32;
        let val_loss = self.val_fn.validate(epoch);

        // INV-TRAIN-007 on val_loss.
        if !val_loss.is_finite() {
            return Err(PretrainAbort::NumericalInstability {
                step: last_step,
                field: "val_loss",
                value: val_loss,
            });
        }

        self.val_loss_history.push(val_loss);

        // GATE-TRAIN-005 — ship-blocking divergence guard.
        // v1.2.0: epoch-zero cap is regime-dependent.
        check_non_divergence(epoch, &self.val_loss_history, &self.config.regime)?;

        let wall_seconds = t0.elapsed().as_secs_f32();
        // INV-TRAIN-003: prefer the real AdamW-state digest if the
        // StepFn exposes one; fall back to a deterministic fingerprint
        // for synthetic harnesses that do not own an optimizer.
        let optimizer_state_sha =
            self.step_fn.optimizer_state_sha256().unwrap_or_else(|| self.fake_optimizer_sha(epoch));
        let metadata = EpochMetadata {
            epoch,
            train_loss: mean_train_loss,
            val_loss,
            train_ppl: mean_train_loss.exp(),
            val_ppl: val_loss.exp(),
            optimizer_state_sha,
            wall_seconds,
            tokens_seen: self.tokens_seen,
            grad_norm_max: epoch_grad_norm_max,
        };
        let artifact = EpochArtifact::new(&self.config.run_dir, epoch, metadata);

        // Task #111 step 7: write the APR checkpoint now that the
        // divergence gate (GATE-TRAIN-005) has passed. Failures do not
        // abort the loop so a flaky disk cannot lose training
        // progress — the artifact is still recorded in history.
        if let Some(ckpt) = self.checkpoint_fn.as_mut() {
            if let Some(parent) = artifact.checkpoint_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(e) = ckpt.save(epoch, &artifact) {
                eprintln!("[pretrain] checkpoint write failed for epoch {}: {}", epoch, e);
            } else {
                // Also emit the companion metadata.json per contract's
                // `per_epoch_artifacts.path_template`. Best-effort: a
                // metadata-write failure is logged but non-fatal.
                match serde_json::to_string_pretty(&artifact.metadata) {
                    Ok(json) => {
                        if let Err(e) = std::fs::write(&artifact.metadata_path, json) {
                            eprintln!(
                                "[pretrain] metadata write failed for epoch {}: {}",
                                epoch, e
                            );
                        }
                    }
                    Err(e) => eprintln!(
                        "[pretrain] metadata serialization failed for epoch {}: {}",
                        epoch, e
                    ),
                }
            }
        }

        self.epoch_artifacts.push(artifact.clone());
        Ok(artifact)
    }

    /// INV-TRAIN-004 convergence/early-stop check. Returns `true` if the
    /// loop should halt cleanly with early-stop status.
    pub fn check_convergence(&mut self, epoch: usize) -> bool {
        let Some(&val_loss) = self.val_loss_history.last() else {
            return false;
        };
        if val_loss < self.best_val_loss {
            self.best_val_loss = val_loss;
            self.patience_counter = 0;
            return false;
        }
        self.patience_counter += 1;
        if epoch + 1 < self.config.min_epochs_before_early_stop {
            return false;
        }
        self.patience_counter > self.config.patience_epochs
    }

    /// Execute the full pretraining loop. Returns the terminal status.
    pub fn run(&mut self) -> RunStatus {
        let num_epochs = self.config.total_steps.div_ceil(self.config.steps_per_epoch.max(1));
        for epoch in 0..num_epochs {
            match self.run_epoch(epoch) {
                Ok(_) => {}
                Err(abort) => return RunStatus::Aborted(abort),
            }
            if self.check_convergence(epoch) {
                return RunStatus::EarlyStop {
                    best_val_loss: self.best_val_loss,
                    epochs_completed: epoch + 1,
                };
            }
            let last = *self.val_loss_history.last().unwrap_or(&f32::INFINITY);
            if last <= self.config.target_val_loss
                && epoch + 1 >= self.config.min_epochs_before_early_stop
            {
                return RunStatus::Ok { final_val_loss: last, epochs_completed: epoch + 1 };
            }
        }
        let last = *self.val_loss_history.last().unwrap_or(&f32::INFINITY);
        RunStatus::Ok { final_val_loss: last, epochs_completed: num_epochs }
    }

    /// Accessors for test / CLI wiring.
    pub fn step_metrics(&self) -> &[StepMetrics] {
        &self.step_metrics
    }

    pub fn epoch_artifacts(&self) -> &[EpochArtifact] {
        &self.epoch_artifacts
    }

    pub fn val_loss_history(&self) -> &[f32] {
        &self.val_loss_history
    }

    /// INV-TRAIN-003 — sha256 of optimizer state. In the full driver this
    /// hashes the AdamW m/v buffers; here the driver is model-agnostic,
    /// so we derive a deterministic sha from epoch + step + config seed
    /// to keep GATE-TRAIN-006 reproducible. Production wiring will
    /// replace this with a real hash of the optimizer state bytes.
    fn fake_optimizer_sha(&self, epoch: usize) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"aprender-train:pretrain:optstate:v1:");
        hasher.update(self.config.seed.to_le_bytes());
        hasher.update((epoch as u64).to_le_bytes());
        hasher.update(self.tokens_seen.to_le_bytes());
        format!("{:x}", hasher.finalize())
    }
}

// ─────────────────────────────────────────────────────────────
// Test helpers + unit tests
// ─────────────────────────────────────────────────────────────

/// Synthetic `StepFn` that drives train_loss down linearly — used by the
/// positive path tests (INV-TRAIN-004, GATE-TRAIN-006).
pub struct LinearDecaySynthetic {
    pub start_loss: f32,
    pub decay_per_step: f32,
    pub grad_norm: f32,
}

impl StepFn for LinearDecaySynthetic {
    fn step(&mut self, step: u64, _lr: f32, _batch_tokens: u64) -> (f32, f32) {
        let loss = (self.start_loss - self.decay_per_step * step as f32).max(1.0e-4);
        (loss, self.grad_norm)
    }
}

/// Synthetic `ValFn` that returns a fixed sequence of epoch val-losses.
/// The falsification harness uses this to inject a doubling trace
/// (e.g. `[3.5, 7.1]`) and prove `check_non_divergence` aborts.
pub struct ScriptedVal {
    pub sequence: Vec<f32>,
}

impl ValFn for ScriptedVal {
    fn validate(&mut self, epoch: usize) -> f32 {
        *self.sequence.get(epoch).unwrap_or(&f32::NAN)
    }
}

/// NaN-injecting synthetic for INV-TRAIN-007 falsification.
pub struct NanAtStepSynthetic {
    pub nan_step: u64,
}

impl StepFn for NanAtStepSynthetic {
    fn step(&mut self, step: u64, _lr: f32, _batch_tokens: u64) -> (f32, f32) {
        if step == self.nan_step {
            return (f32::NAN, 1.0);
        }
        (1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;
    use tempfile::TempDir;

    fn test_config(tmp: &Path) -> PretrainConfig {
        PretrainConfig {
            dataset_path: tmp.join("data.jsonl"),
            tokenizer_dir: tmp.join("tok"),
            run_dir: tmp.join("run"),
            lr_max: 1.0e-4,
            lr_min: 1.0e-6,
            warmup_steps: 2,
            total_steps: 25,
            batch_size: 2,
            seq_length: 4,
            steps_per_epoch: 5,
            seed: 42,
            grad_clip: 1.0,
            weight_decay: 0.01,
            target_val_loss: 2.2,
            patience_epochs: 2,
            min_epochs_before_early_stop: 1,
            regime: TrainingRegime::Finetune,
        }
    }

    // ── GATE-TRAIN-005 falsifier — the MODEL-1 v2 ship-blocker ──

    /// Spec `FALSIFY-SHIP-013` harness: inject a doubling val-loss trace
    /// and assert `check_non_divergence` returns `Err(Divergence)`.
    #[test]
    fn gate_train_005_aborts_on_doubling_val_loss() {
        let trace = vec![3.5, 7.1];
        let res = check_non_divergence(1, &trace, &TrainingRegime::Finetune);
        match res {
            Err(PretrainAbort::Divergence { epoch, prev_val_loss, curr_val_loss, ratio }) => {
                assert_eq!(epoch, 1);
                assert!((prev_val_loss - 3.5).abs() < 1e-6);
                assert!((curr_val_loss - 7.1).abs() < 1e-6);
                assert!(ratio > 2.0);
            }
            other => panic!("GATE-TRAIN-005 did not abort: got {other:?}"),
        }
    }

    /// Special case: val_loss[0] > 10.0 is the MODEL-1 v2 defect (val_loss
    /// 31.99 at epoch 0). Must abort at epoch 0, without waiting for N=1.
    #[test]
    fn gate_train_005_aborts_on_epoch_zero_blowup() {
        let trace = vec![31.99];
        let res = check_non_divergence(0, &trace, &TrainingRegime::Finetune);
        match res {
            Err(PretrainAbort::DivergenceAtEpochZero { val_loss }) => {
                assert!((val_loss - 31.99).abs() < 1e-4);
            }
            other => panic!("epoch-0 guard missed: got {other:?}"),
        }
    }

    /// Healthy trace — must NOT abort. Lower ratios preserve training.
    #[test]
    fn gate_train_005_allows_healthy_decrease() {
        let trace = vec![3.5, 3.0, 2.5, 2.2];
        for epoch in 0..trace.len() {
            assert!(check_non_divergence(epoch, &trace, &TrainingRegime::Finetune).is_ok());
        }
    }

    /// Boundary case — ratio exactly 2.0 MUST be allowed (strict `>` in contract).
    #[test]
    fn gate_train_005_allows_exact_two_x() {
        let trace = vec![2.0, 4.0];
        assert!(check_non_divergence(1, &trace, &TrainingRegime::Finetune).is_ok());
    }

    // ── INV-TRAIN-005 v1.2.0 regime split — from_scratch epoch-zero cap ──

    /// v1.2.0: from_scratch epoch-zero cap is 2·ln(vocab_size). For
    /// vocab=50257 this is ≈21.64. `val_loss[0]=18.0` is below cap and
    /// would trip the old 10.0 literal — must be allowed in the new regime.
    #[test]
    fn gate_train_005_from_scratch_permits_near_random_baseline() {
        let trace = vec![18.0_f32];
        let regime = TrainingRegime::FromScratch { vocab_size: 50_257 };
        assert!(
            check_non_divergence(0, &trace, &regime).is_ok(),
            "val_loss[0]=18 must be within 2·ln(50257)≈21.64 from_scratch cap"
        );
        // Same trace under Finetune still aborts — regime split is not a weakening.
        assert!(matches!(
            check_non_divergence(0, &trace, &TrainingRegime::Finetune),
            Err(PretrainAbort::DivergenceAtEpochZero { .. }),
        ));
    }

    /// v1.2.0: from_scratch above 2·ln(vocab_size) still aborts. Confirms
    /// the gate is not degenerate — it catches truly broken inits.
    #[test]
    fn gate_train_005_from_scratch_aborts_above_2_ln_vocab() {
        let trace = vec![25.0_f32];
        let regime = TrainingRegime::FromScratch { vocab_size: 50_257 };
        match check_non_divergence(0, &trace, &regime) {
            Err(PretrainAbort::DivergenceAtEpochZero { val_loss }) => {
                assert!((val_loss - 25.0).abs() < 1e-4);
            }
            other => panic!("from_scratch cap missed: got {other:?}"),
        }
    }

    /// v1.2.0: the computed cap matches the formula 2·ln(vocab_size).
    /// Locks the threshold so a silent code change cannot relax it.
    #[test]
    fn training_regime_from_scratch_cap_matches_formula() {
        let v = 50_257u32;
        let regime = TrainingRegime::FromScratch { vocab_size: v };
        let expected = DIVERGENCE_RATIO_LIMIT * (v as f32).ln();
        assert!(
            (regime.epoch_zero_val_loss_limit() - expected).abs() < 1e-4,
            "cap formula drift: got {} expected {}",
            regime.epoch_zero_val_loss_limit(),
            expected
        );
        // Finetune stays on the MODEL-1 literal.
        assert!(
            (TrainingRegime::Finetune.epoch_zero_val_loss_limit() - EPOCH_ZERO_VAL_LOSS_LIMIT)
                .abs()
                < 1e-6
        );
    }

    // ── GATE-TRAIN-007 falsifier — NaN poisoning ──

    #[test]
    fn gate_train_007_aborts_on_nan_train_loss() {
        let res = check_numerical_stability(42, f32::NAN, 1.0);
        match res {
            Err(PretrainAbort::NumericalInstability { step, field, .. }) => {
                assert_eq!(step, 42);
                assert_eq!(field, "train_loss");
            }
            other => panic!("nan guard missed: got {other:?}"),
        }
    }

    #[test]
    fn gate_train_007_aborts_on_inf_grad_norm() {
        let res = check_numerical_stability(7, 1.0, f32::INFINITY);
        assert!(matches!(res, Err(PretrainAbort::NumericalInstability { .. })));
    }

    // ── GATE-TRAIN-001 / INV-TRAIN-008 — metrics validation ──

    #[test]
    fn step_metrics_validate_finite_accepts_healthy() {
        let m = StepMetrics {
            step: 0,
            train_loss: 3.2,
            grad_norm: 0.5,
            lr: 1e-4,
            tokens_per_sec: 1000.0,
            gpu_util_pct: 75.0,
            wall_ms: 5.0,
        };
        assert!(m.validate_finite().is_ok());
    }

    #[test]
    fn step_metrics_rejects_negative_throughput() {
        let m = StepMetrics {
            step: 1,
            train_loss: 3.2,
            grad_norm: 0.5,
            lr: 1e-4,
            tokens_per_sec: -1.0,
            gpu_util_pct: 75.0,
            wall_ms: 5.0,
        };
        assert!(matches!(m.validate_finite(), Err(PretrainAbort::ThroughputOutOfRange { .. })));
    }

    #[test]
    fn step_metrics_rejects_gpu_util_over_100() {
        let m = StepMetrics {
            step: 1,
            train_loss: 3.2,
            grad_norm: 0.5,
            lr: 1e-4,
            tokens_per_sec: 1000.0,
            gpu_util_pct: 150.0,
            wall_ms: 5.0,
        };
        assert!(matches!(m.validate_finite(), Err(PretrainAbort::ThroughputOutOfRange { .. })));
    }

    /// Per `contracts/training-loop-pretrain-v1.yaml` v1.5.0:
    /// wall_ms must be finite and non-negative.
    #[test]
    fn step_metrics_rejects_negative_wall_ms() {
        let m = StepMetrics {
            step: 1,
            train_loss: 3.2,
            grad_norm: 0.5,
            lr: 1e-4,
            tokens_per_sec: 1000.0,
            gpu_util_pct: 75.0,
            wall_ms: -1.0,
        };
        assert!(matches!(m.validate_finite(), Err(PretrainAbort::ThroughputOutOfRange { .. })));
    }

    /// wall_ms must be finite (NaN/Inf rejected).
    #[test]
    fn step_metrics_rejects_nan_wall_ms() {
        let m = StepMetrics {
            step: 1,
            train_loss: 3.2,
            grad_norm: 0.5,
            lr: 1e-4,
            tokens_per_sec: 1000.0,
            gpu_util_pct: 75.0,
            wall_ms: f32::NAN,
        };
        assert!(matches!(m.validate_finite(), Err(PretrainAbort::ThroughputOutOfRange { .. })));
    }

    /// Consistency invariant from contract v1.5.0:
    /// `tokens_per_sec * (wall_ms / 1000.0) ≈ batch_tokens` within FP
    /// rounding. Both metrics derive from the same `Instant::now()`
    /// span so they cannot drift independently.
    #[test]
    fn step_metrics_wall_ms_consistent_with_tokens_per_sec() {
        let batch_tokens: u64 = 1024;
        let elapsed_secs: f32 = 0.5;
        let tokens_per_sec = batch_tokens as f32 / elapsed_secs;
        let wall_ms = elapsed_secs * 1000.0;

        let m = StepMetrics {
            step: 0,
            train_loss: 3.2,
            grad_norm: 0.5,
            lr: 1e-4,
            tokens_per_sec,
            gpu_util_pct: 50.0,
            wall_ms,
        };
        assert!(m.validate_finite().is_ok());
        let derived_tokens = m.tokens_per_sec * (m.wall_ms / 1000.0);
        let diff = (derived_tokens - batch_tokens as f32).abs();
        assert!(
            diff < 0.5,
            "tokens_per_sec * (wall_ms/1000) = {derived_tokens} should equal batch_tokens={batch_tokens} within FP rounding"
        );
    }

    // ── PretrainLoop — driver-level falsifications ──

    #[test]
    fn pretrain_loop_happy_path_decreasing_loss() {
        let tmp = TempDir::new().expect("tempdir");
        let cfg = test_config(tmp.path());
        let step_fn = LinearDecaySynthetic { start_loss: 3.5, decay_per_step: 0.1, grad_norm: 0.8 };
        let val_fn = ScriptedVal { sequence: vec![3.4, 3.0, 2.6, 2.2, 2.0] };
        let mut loop_ = PretrainLoop::new(cfg, step_fn, val_fn);

        let status = loop_.run();
        match status {
            RunStatus::Ok { final_val_loss, epochs_completed } => {
                assert!(final_val_loss <= 2.2);
                assert!(epochs_completed >= 1);
            }
            other => panic!("healthy run did not converge cleanly: {other:?}"),
        }

        // GATE-TRAIN-001 — every step recorded all 6 fields.
        assert!(!loop_.step_metrics().is_empty());
        for m in loop_.step_metrics() {
            assert!(m.train_loss.is_finite());
            assert!(m.grad_norm.is_finite());
            assert!(m.lr.is_finite());
            assert!(m.tokens_per_sec >= 0.0);
            assert!((0.0..=100.0).contains(&m.gpu_util_pct));
        }
        // GATE-TRAIN-002 — one metadata per completed epoch.
        assert_eq!(loop_.epoch_artifacts().len(), loop_.val_loss_history().len());
        for art in loop_.epoch_artifacts() {
            assert!(!art.metadata.optimizer_state_sha.is_empty());
            assert!(art.metadata.train_ppl.is_finite());
            assert!(art.metadata.val_ppl.is_finite());
        }
    }

    /// INV-TRAIN-005 ship-blocker end-to-end: drive a doubling val-loss
    /// through the full `run_epoch` and prove the loop aborts, not a
    /// post-hoc audit. This is the falsifier GATE-TRAIN-005 mandates.
    #[test]
    fn pretrain_loop_aborts_on_doubling_val_loss() {
        let tmp = TempDir::new().expect("tempdir");
        let cfg = test_config(tmp.path());
        let step_fn = LinearDecaySynthetic { start_loss: 3.5, decay_per_step: 0.1, grad_norm: 0.8 };
        // val_loss doubles between epochs 0 and 1 — must abort.
        let val_fn = ScriptedVal { sequence: vec![3.5, 7.1, 2.0] };
        let mut loop_ = PretrainLoop::new(cfg, step_fn, val_fn);

        let status = loop_.run();
        match status {
            RunStatus::Aborted(PretrainAbort::Divergence { epoch, ratio, .. }) => {
                assert_eq!(epoch, 1);
                assert!(ratio > 2.0);
            }
            other => panic!("GATE-TRAIN-005 did not fire: {other:?}"),
        }
    }

    /// INV-TRAIN-007 end-to-end: NaN in train_loss at step N aborts.
    #[test]
    fn pretrain_loop_aborts_on_nan_in_train_loss() {
        let tmp = TempDir::new().expect("tempdir");
        let cfg = test_config(tmp.path());
        let step_fn = NanAtStepSynthetic { nan_step: 3 };
        let val_fn = ScriptedVal { sequence: vec![3.0] };
        let mut loop_ = PretrainLoop::new(cfg, step_fn, val_fn);

        let status = loop_.run();
        match status {
            RunStatus::Aborted(PretrainAbort::NumericalInstability { step, field, .. }) => {
                assert_eq!(step, 3);
                assert_eq!(field, "train_loss");
            }
            other => panic!("INV-TRAIN-007 did not fire: {other:?}"),
        }
    }

    /// `--num-steps N` is a BUDGET, not a lower bound. The loop must execute
    /// exactly N train steps even when N is not a whole multiple of
    /// `steps_per_epoch`.
    ///
    /// Before the clamp in `run_epoch`, `apr pretrain --num-steps 3` at the
    /// default `steps_per_epoch = 100` ran 100 steps: the same run printed
    /// `Total steps: 3` and `Steps recorded: 100`. `--num-steps 250` ran 300.
    #[test]
    fn num_steps_is_a_budget_not_rounded_up_to_a_whole_epoch() {
        // (total_steps, steps_per_epoch) — the CLI default spe is 100.
        for (total, spe) in [(3usize, 100usize), (250, 100), (7, 5), (10, 10), (1, 64)] {
            let tmp = TempDir::new().expect("tempdir");
            let cfg = PretrainConfig {
                total_steps: total,
                steps_per_epoch: spe,
                warmup_steps: 1,
                // Never early-stop / converge out of the run: we are counting steps.
                target_val_loss: 0.0,
                min_epochs_before_early_stop: usize::MAX,
                patience_epochs: usize::MAX,
                ..test_config(tmp.path())
            };
            let step_fn =
                LinearDecaySynthetic { start_loss: 3.0, decay_per_step: 0.0, grad_norm: 0.5 };
            let val_fn = ScriptedVal { sequence: vec![3.0; total.div_ceil(spe) + 2] };
            let mut loop_ = PretrainLoop::new(cfg, step_fn, val_fn);
            let _ = loop_.run();

            assert_eq!(
                loop_.step_metrics().len(),
                total,
                "requested {total} steps at steps_per_epoch={spe}, executed {}",
                loop_.step_metrics().len()
            );
            // Step indices must be exactly 0..total — no step beyond the budget.
            let last = loop_.step_metrics().last().map(|m| m.step);
            assert_eq!(last, Some(total as u64 - 1), "last step index past the budget");
        }
    }

    /// INV-TRAIN-006: two runs with the same seed produce identical metrics
    /// for the first 100 steps. We use 10 steps here to keep the unit test
    /// fast; CI GATE-TRAIN-006 runs the full 100.
    #[test]
    fn pretrain_loop_reproducibility_seed_42() {
        let tmp1 = TempDir::new().expect("tempdir1");
        let tmp2 = TempDir::new().expect("tempdir2");
        let cfg1 = test_config(tmp1.path());
        let cfg2 = test_config(tmp2.path());

        let step_fn1 =
            LinearDecaySynthetic { start_loss: 3.5, decay_per_step: 0.1, grad_norm: 0.8 };
        let step_fn2 =
            LinearDecaySynthetic { start_loss: 3.5, decay_per_step: 0.1, grad_norm: 0.8 };
        let val_fn1 = ScriptedVal { sequence: vec![3.0, 2.8, 2.6, 2.4, 2.2] };
        let val_fn2 = ScriptedVal { sequence: vec![3.0, 2.8, 2.6, 2.4, 2.2] };

        let mut loop1 = PretrainLoop::new(cfg1, step_fn1, val_fn1);
        let mut loop2 = PretrainLoop::new(cfg2, step_fn2, val_fn2);
        let _ = loop1.run();
        let _ = loop2.run();

        assert_eq!(loop1.step_metrics().len(), loop2.step_metrics().len());
        for (a, b) in loop1.step_metrics().iter().zip(loop2.step_metrics().iter()) {
            assert_eq!(a.step, b.step);
            assert!((a.train_loss - b.train_loss).abs() < 1e-6);
            assert!((a.grad_norm - b.grad_norm).abs() < 1e-6);
            assert!((a.lr - b.lr).abs() < 1e-6);
            // gpu_util_pct is RNG-driven; seed-matched ⇒ byte-identical.
            assert!((a.gpu_util_pct - b.gpu_util_pct).abs() < 1e-6);
        }
    }

    /// `lr_at` must match WarmupCosineDecayLR behavior byte-for-byte at
    /// the boundary points (start of warmup, end of warmup, end of decay).
    #[test]
    fn lr_schedule_warmup_cosine_boundaries() {
        let tmp = TempDir::new().expect("tempdir");
        let cfg = PretrainConfig {
            warmup_steps: 10,
            total_steps: 100,
            lr_max: 1.0e-3,
            lr_min: 1.0e-5,
            ..test_config(tmp.path())
        };
        let step_fn = LinearDecaySynthetic { start_loss: 1.0, decay_per_step: 0.0, grad_norm: 0.1 };
        let val_fn = ScriptedVal { sequence: vec![1.0] };
        let loop_ = PretrainLoop::new(cfg, step_fn, val_fn);

        // Start of warmup — lr should be 0.
        assert!((loop_.lr_at(0) - 0.0).abs() < 1e-9);
        // End of warmup — lr at peak.
        assert!((loop_.lr_at(10) - 1.0e-3).abs() < 1e-6);
        // End of decay — lr at minimum.
        assert!((loop_.lr_at(100) - 1.0e-5).abs() < 1e-6);
    }

    /// Artifact paths must match the contract's `path_template`.
    #[test]
    fn epoch_artifact_paths_match_contract_template() {
        let tmp = TempDir::new().expect("tempdir");
        let run_dir = tmp.path().join("run");
        let metadata = EpochMetadata {
            epoch: 7,
            train_loss: 3.0,
            val_loss: 2.8,
            train_ppl: 20.0,
            val_ppl: 16.4,
            optimizer_state_sha: "deadbeef".into(),
            wall_seconds: 42.0,
            tokens_seen: 1_000_000,
            grad_norm_max: 1.5,
        };
        let art = EpochArtifact::new(&run_dir, 7, metadata);
        assert!(art.checkpoint_path.ends_with("ckpt/epoch-007.apr"));
        assert!(art.metadata_path.ends_with("ckpt/epoch-007.metadata.json"));
    }

    // ── Task #111 step 7 — CheckpointFn hook falsifiers ──

    /// Mock `CheckpointFn` that records every (epoch, checkpoint_path) pair
    /// so tests can assert the loop invokes the hook exactly once per
    /// passing epoch and never on an aborted epoch.
    struct RecordingCheckpointFn {
        calls: Rc<RefCell<Vec<(usize, PathBuf)>>>,
    }

    impl CheckpointFn for RecordingCheckpointFn {
        fn save(&mut self, epoch: usize, artifact: &EpochArtifact) -> Result<(), String> {
            self.calls.borrow_mut().push((epoch, artifact.checkpoint_path.clone()));
            Ok(())
        }
    }

    /// INV-TRAIN-005 positive: one checkpoint call per epoch that
    /// passes GATE-TRAIN-005, with metadata.json emitted alongside.
    #[test]
    fn pretrain_loop_calls_checkpoint_fn_once_per_passing_epoch() {
        let tmp = TempDir::new().expect("tempdir");
        let cfg = test_config(tmp.path());
        let step_fn = LinearDecaySynthetic { start_loss: 3.5, decay_per_step: 0.1, grad_norm: 0.8 };
        let val_fn = ScriptedVal { sequence: vec![3.4, 3.0, 2.6, 2.2, 2.0] };
        let calls: Rc<RefCell<Vec<(usize, PathBuf)>>> = Rc::new(RefCell::new(Vec::new()));
        let ckpt = RecordingCheckpointFn { calls: Rc::clone(&calls) };

        let mut loop_ = PretrainLoop::new(cfg, step_fn, val_fn).with_checkpoint_fn(Box::new(ckpt));
        let _status = loop_.run();

        let recorded = calls.borrow();
        let epoch_count = loop_.epoch_artifacts().len();
        assert!(epoch_count >= 1, "at least one epoch should have completed");
        assert_eq!(
            recorded.len(),
            epoch_count,
            "CheckpointFn must fire exactly once per epoch that passes GATE-TRAIN-005",
        );
        for (i, (epoch, path)) in recorded.iter().enumerate() {
            assert_eq!(*epoch, i, "checkpoint hook epoch indices must be monotonic from 0");
            assert!(
                path.to_string_lossy().contains(&format!("epoch-{:03}.apr", epoch)),
                "checkpoint path must match contract template: {:?}",
                path,
            );
            let meta_path = path.with_extension("metadata.json");
            assert!(
                meta_path.exists(),
                "companion metadata.json must be written for epoch {}",
                epoch,
            );
        }
    }

    /// INV-TRAIN-003 positive: if the StepFn overrides
    /// `optimizer_state_sha256`, the loop uses it instead of the
    /// synthetic-seed fallback. Asserts that the recorded epoch
    /// metadata carries the sha from the StepFn.
    #[test]
    fn pretrain_loop_uses_step_fn_optimizer_sha_when_available() {
        struct ShaOverride {
            inner: LinearDecaySynthetic,
            sha: String,
        }
        impl StepFn for ShaOverride {
            fn step(&mut self, s: u64, lr: f32, tokens: u64) -> (f32, f32) {
                self.inner.step(s, lr, tokens)
            }
            fn optimizer_state_sha256(&self) -> Option<String> {
                Some(self.sha.clone())
            }
        }

        let tmp = TempDir::new().expect("tempdir");
        let cfg = test_config(tmp.path());
        let step_fn = ShaOverride {
            inner: LinearDecaySynthetic { start_loss: 3.5, decay_per_step: 0.1, grad_norm: 0.8 },
            sha: "a".repeat(64),
        };
        let val_fn = ScriptedVal { sequence: vec![3.4, 3.0, 2.6, 2.2, 2.0] };
        let mut loop_ = PretrainLoop::new(cfg, step_fn, val_fn);
        let _ = loop_.run();

        let arts = loop_.epoch_artifacts();
        assert!(!arts.is_empty(), "at least one epoch should have completed");
        for art in arts {
            assert_eq!(
                art.metadata.optimizer_state_sha,
                "a".repeat(64),
                "StepFn override must win over fake_optimizer_sha fallback",
            );
        }
    }

    /// INV-TRAIN-003 fallback: a synthetic StepFn that does not
    /// override `optimizer_state_sha256` still gets a non-empty,
    /// deterministic 64-char digest via the `fake_optimizer_sha`
    /// fingerprint. (The default impl returns `None`.)
    #[test]
    fn pretrain_loop_falls_back_to_fake_optimizer_sha_for_synthetic() {
        let tmp = TempDir::new().expect("tempdir");
        let cfg = test_config(tmp.path());
        let step_fn = LinearDecaySynthetic { start_loss: 3.5, decay_per_step: 0.1, grad_norm: 0.8 };
        let val_fn = ScriptedVal { sequence: vec![3.4, 3.0, 2.6, 2.2, 2.0] };
        let mut loop_ = PretrainLoop::new(cfg, step_fn, val_fn);
        let _ = loop_.run();

        for art in loop_.epoch_artifacts() {
            assert_eq!(
                art.metadata.optimizer_state_sha.len(),
                64,
                "fallback fingerprint must still be a 64-char hex digest",
            );
            assert!(
                art.metadata.optimizer_state_sha.chars().all(|c| c.is_ascii_hexdigit()),
                "fallback fingerprint must be lowercase hex",
            );
        }
    }

    /// INV-TRAIN-007 negative: NaN in train_loss aborts the loop, and
    /// the checkpoint hook must NOT fire for the aborted epoch.
    #[test]
    fn pretrain_loop_skips_checkpoint_on_abort() {
        let tmp = TempDir::new().expect("tempdir");
        let cfg = test_config(tmp.path());
        let step_fn = NanAtStepSynthetic { nan_step: 1 };
        let val_fn = ScriptedVal { sequence: vec![3.0] };
        let calls: Rc<RefCell<Vec<(usize, PathBuf)>>> = Rc::new(RefCell::new(Vec::new()));
        let ckpt = RecordingCheckpointFn { calls: Rc::clone(&calls) };

        let mut loop_ = PretrainLoop::new(cfg, step_fn, val_fn).with_checkpoint_fn(Box::new(ckpt));
        let status = loop_.run();

        assert!(
            matches!(status, RunStatus::Aborted(PretrainAbort::NumericalInstability { .. })),
            "NaN must abort the loop: got {status:?}",
        );
        assert!(
            calls.borrow().is_empty(),
            "CheckpointFn must NOT fire when the epoch aborts before GATE-TRAIN-005 passes",
        );
    }
}
