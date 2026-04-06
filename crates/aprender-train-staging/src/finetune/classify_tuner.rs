//! Automatic hyperparameter tuning for classification fine-tuning (SPEC-TUNE-2026-001)
//!
//! Provides `ClassifyTuner` which orchestrates HPO search over LoRA + classifier
//! configurations using existing TPE/Grid/Hyperband infrastructure.
//!
//! # Architecture
//!
//! ```text
//! ClassifyTuner
//!   ├── TuneSearcher (TPE / Grid / Random)
//!   ├── TuneScheduler (ASHA / Median / None)
//!   └── per trial: ClassifyPipeline → ClassifyTrainer → EpochMetrics
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::optim::{HyperparameterSpace, ParameterDomain, ParameterValue};

// Re-export searcher/scheduler types from submodule
pub use super::tune_searchers::{
    AshaScheduler, GridSearcher, MedianScheduler, NoScheduler, RandomSearcher, TpeSearcher,
    TuneScheduler, TuneSearcher,
};

// ═══════════════════════════════════════════════════════════════════════
// Configuration and result types
// ═══════════════════════════════════════════════════════════════════════

/// Tuning strategy selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TuneStrategy {
    Tpe,
    Grid,
    Random,
}

impl std::str::FromStr for TuneStrategy {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "tpe" | "bayesian" => Ok(Self::Tpe),
            "grid" => Ok(Self::Grid),
            "random" => Ok(Self::Random),
            _ => Err(format!("Unknown strategy: {s}. Use: tpe, grid, random")),
        }
    }
}

impl std::fmt::Display for TuneStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tpe => write!(f, "tpe"),
            Self::Grid => write!(f, "grid"),
            Self::Random => write!(f, "random"),
        }
    }
}

/// Scheduler selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulerKind {
    Asha,
    Median,
    None,
}

impl std::str::FromStr for SchedulerKind {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "asha" => Ok(Self::Asha),
            "median" => Ok(Self::Median),
            "none" => Ok(Self::None),
            _ => Err(format!("Unknown scheduler: {s}. Use: asha, median, none")),
        }
    }
}

/// Tuning run configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuneConfig {
    /// Maximum number of trials.
    pub budget: usize,
    /// Search strategy (TPE, Grid, Random).
    pub strategy: TuneStrategy,
    /// Scheduler for early stopping.
    pub scheduler: SchedulerKind,
    /// Scout mode: 1 epoch per trial, no scheduling.
    pub scout: bool,
    /// Maximum epochs per trial (full mode).
    pub max_epochs: usize,
    /// Number of output classes.
    pub num_classes: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Optional time limit in seconds.
    pub time_limit_secs: Option<u64>,
}

impl Default for TuneConfig {
    fn default() -> Self {
        Self {
            budget: 10,
            strategy: TuneStrategy::Tpe,
            scheduler: SchedulerKind::Asha,
            scout: false,
            max_epochs: 20,
            num_classes: 5,
            seed: 42,
            time_limit_secs: None,
        }
    }
}

/// Summary of a single completed trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialSummary {
    /// Trial index.
    pub id: usize,
    /// Validation loss (best epoch).
    pub val_loss: f64,
    /// Validation accuracy (best epoch).
    pub val_accuracy: f64,
    /// Training loss (final epoch).
    pub train_loss: f64,
    /// Training accuracy (final epoch).
    pub train_accuracy: f64,
    /// Number of epochs actually run.
    pub epochs_run: usize,
    /// Wall-clock time in milliseconds.
    pub time_ms: u64,
    /// Hyperparameter configuration.
    pub config: HashMap<String, ParameterValue>,
    /// Trial status.
    pub status: String,
}

/// Result of a complete tuning run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuneResult {
    /// Strategy used.
    pub strategy: String,
    /// Mode (scout or full).
    pub mode: String,
    /// Budget (total trials attempted).
    pub budget: usize,
    /// All trial summaries, sorted by val_loss ascending (best first).
    pub trials: Vec<TrialSummary>,
    /// ID of the best trial.
    pub best_trial_id: usize,
    /// Total wall-clock time in milliseconds.
    pub total_time_ms: u64,
}

/// Build the default 9-parameter search space for classification HPO.
///
/// Parameters from SPEC-TUNE-2026-001 §4.1.
pub fn default_classify_search_space() -> HyperparameterSpace {
    let mut space = HyperparameterSpace::new();

    // Learning rate: 5e-6 .. 5e-4 (log-scale)
    space.add(
        "learning_rate",
        ParameterDomain::Continuous { low: 5e-6, high: 5e-4, log_scale: true },
    );

    // LoRA rank: 4 .. 64 (discrete, step of 4 → 4,8,12,...,64)
    space.add("lora_rank", ParameterDomain::Discrete { low: 1, high: 16 });

    // Alpha ratio: 0.5 .. 2.0 (ties alpha to rank)
    space.add(
        "lora_alpha_ratio",
        ParameterDomain::Continuous { low: 0.5, high: 2.0, log_scale: false },
    );

    // Batch size: categorical [8, 16, 32, 64, 128]
    space.add(
        "batch_size",
        ParameterDomain::Categorical {
            choices: vec![
                "8".to_string(),
                "16".to_string(),
                "32".to_string(),
                "64".to_string(),
                "128".to_string(),
            ],
        },
    );

    // Warmup fraction: 0.01 .. 0.2
    space.add(
        "warmup_fraction",
        ParameterDomain::Continuous { low: 0.01, high: 0.2, log_scale: false },
    );

    // Gradient clip norm: 0.5 .. 5.0
    space.add(
        "gradient_clip_norm",
        ParameterDomain::Continuous { low: 0.5, high: 5.0, log_scale: false },
    );

    // Class weights strategy
    space.add(
        "class_weights",
        ParameterDomain::Categorical {
            choices: vec![
                "uniform".to_string(),
                "inverse_freq".to_string(),
                "sqrt_inverse".to_string(),
            ],
        },
    );

    // Target modules
    space.add(
        "target_modules",
        ParameterDomain::Categorical {
            choices: vec!["qv".to_string(), "qkv".to_string(), "all_linear".to_string()],
        },
    );

    // LR min ratio (cosine decay floor = lr * ratio)
    space.add(
        "lr_min_ratio",
        ParameterDomain::Continuous { low: 0.001, high: 0.1, log_scale: true },
    );

    space
}

/// Convert a trial's ParameterValue map into concrete hyperparameter values.
///
/// Returns (learning_rate, lora_rank, lora_alpha, batch_size, warmup_fraction,
///          gradient_clip_norm, class_weights_strategy, target_modules, lr_min_ratio).
#[allow(clippy::implicit_hasher)]
pub fn extract_trial_params(
    config: &HashMap<String, ParameterValue>,
) -> (f32, usize, f32, usize, f32, f32, String, String, f32) {
    let lr = config.get("learning_rate").and_then(ParameterValue::as_float).unwrap_or(1e-4) as f32;

    // lora_rank: discrete 1-16 maps to rank * 4 → 4,8,...,64
    let rank_raw = config.get("lora_rank").and_then(ParameterValue::as_int).unwrap_or(4) as usize;
    let rank = (rank_raw * 4).clamp(4, 64);

    let alpha_ratio =
        config.get("lora_alpha_ratio").and_then(ParameterValue::as_float).unwrap_or(1.0) as f32;
    let alpha = rank as f32 * alpha_ratio;

    let batch_size = config
        .get("batch_size")
        .and_then(ParameterValue::as_str)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(32);

    let warmup =
        config.get("warmup_fraction").and_then(ParameterValue::as_float).unwrap_or(0.1) as f32;

    let clip =
        config.get("gradient_clip_norm").and_then(ParameterValue::as_float).unwrap_or(1.0) as f32;

    let weights_strategy = config
        .get("class_weights")
        .and_then(ParameterValue::as_str)
        .unwrap_or("uniform")
        .to_string();

    let targets =
        config.get("target_modules").and_then(ParameterValue::as_str).unwrap_or("qv").to_string();

    let lr_min_ratio =
        config.get("lr_min_ratio").and_then(ParameterValue::as_float).unwrap_or(0.01) as f32;

    (lr, rank, alpha, batch_size, warmup, clip, weights_strategy, targets, lr_min_ratio)
}

// ═══════════════════════════════════════════════════════════════════════
// ClassifyTuner
// ═══════════════════════════════════════════════════════════════════════

/// Orchestrates hyperparameter optimization for classification fine-tuning.
///
/// Coordinates:
/// 1. Searcher (TPE/Grid/Random) to suggest configs
/// 2. Scheduler (ASHA/Median/None) for early stopping
/// 3. ClassifyTrainer execution per trial
/// 4. Leaderboard ranking and persistence
#[derive(Debug)]
pub struct ClassifyTuner {
    /// Tuning configuration.
    pub config: TuneConfig,
    /// Search space.
    pub space: HyperparameterSpace,
    /// Completed trial summaries.
    pub leaderboard: Vec<TrialSummary>,
}

impl ClassifyTuner {
    /// Create a new tuner with the default classification search space.
    pub fn new(config: TuneConfig) -> crate::Result<Self> {
        if config.budget == 0 {
            return Err(crate::Error::ConfigError(
                "FALSIFY-TUNE-001: budget must be > 0".to_string(),
            ));
        }
        if config.num_classes == 0 {
            return Err(crate::Error::ConfigError(
                "FALSIFY-TUNE-004: num_classes must be > 0".to_string(),
            ));
        }

        let space = default_classify_search_space();

        Ok(Self { config, space, leaderboard: Vec::new() })
    }

    /// Create the appropriate searcher based on strategy.
    pub fn build_searcher(&self) -> Box<dyn TuneSearcher> {
        let n_startup = (self.config.budget / 3).max(3);
        match self.config.strategy {
            TuneStrategy::Tpe => Box::new(TpeSearcher::new(self.space.clone(), n_startup)),
            TuneStrategy::Grid => Box::new(GridSearcher::new(self.space.clone(), 3)),
            TuneStrategy::Random => Box::new(RandomSearcher::new(self.space.clone())),
        }
    }

    /// Create the appropriate scheduler.
    pub fn build_scheduler(&self) -> Box<dyn TuneScheduler> {
        if self.config.scout {
            return Box::new(NoScheduler);
        }
        match self.config.scheduler {
            SchedulerKind::Asha => Box::new(AshaScheduler::new(1, 3.0)),
            SchedulerKind::Median => Box::new(MedianScheduler::new(1)),
            SchedulerKind::None => Box::new(NoScheduler),
        }
    }

    /// Record a completed trial result and update the leaderboard.
    pub fn record_trial(&mut self, summary: TrialSummary) {
        self.leaderboard.push(summary);
        // Sort leaderboard by val_loss ascending (best first)
        self.leaderboard.sort_by(|a, b| {
            a.val_loss.partial_cmp(&b.val_loss).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get the best trial summary from the leaderboard.
    pub fn best_trial(&self) -> Option<&TrialSummary> {
        self.leaderboard.first()
    }

    /// Build the final TuneResult from collected trials.
    pub fn into_result(self, total_time_ms: u64) -> TuneResult {
        let best_id = self.leaderboard.first().map_or(0, |t| t.id);
        TuneResult {
            strategy: self.config.strategy.to_string(),
            mode: if self.config.scout { "scout".to_string() } else { "full".to_string() },
            budget: self.config.budget,
            trials: self.leaderboard,
            best_trial_id: best_id,
            total_time_ms,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[path = "classify_tuner_tests.rs"]
mod tests;
