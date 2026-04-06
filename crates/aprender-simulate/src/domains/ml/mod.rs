//! Machine Learning Simulation Engine.
//!
//! Provides deterministic, reproducible simulation of ML workflows using
//! Popperian falsification methodology. Implements TPS principles:
//! - Jidoka: Stop-on-anomaly detection
//! - Heijunka: Load-balanced batch processing
//! - Kaizen: Continuous improvement via feedback
//!
//! # Example
//!
//! ```rust
//! use simular::domains::ml::{TrainingSimulation, TrainingConfig, AnomalyDetector};
//! use simular::engine::rng::SimRng;
//!
//! let mut sim = TrainingSimulation::new(42);
//! let config = TrainingConfig::default();
//! // Training simulation would run here
//! ```

pub mod jidoka;
pub mod multi_turn;
pub mod prediction;

#[cfg(test)]
mod tests;

pub use jidoka::*;
pub use multi_turn::*;
pub use prediction::*;

use serde::{Deserialize, Serialize};

use crate::engine::rng::{RngState, SimRng};
use crate::engine::SimTime;
use crate::error::{SimError, SimResult};
use crate::replay::EventJournal;

// ============================================================================
// Training Simulation Types
// ============================================================================

/// Training hyperparameters configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate.
    pub learning_rate: f64,
    /// Batch size for training.
    pub batch_size: usize,
    /// Number of epochs.
    pub epochs: u64,
    /// Early stopping patience (None = disabled).
    pub early_stopping: Option<usize>,
    /// Gradient clipping max norm (None = disabled).
    pub gradient_clip: Option<f64>,
    /// Weight decay (L2 regularization).
    pub weight_decay: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            early_stopping: Some(10),
            gradient_clip: Some(1.0),
            weight_decay: 0.0001,
        }
    }
}

/// Training state captured at each epoch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    /// Current epoch.
    pub epoch: u64,
    /// Training loss.
    pub loss: f64,
    /// Validation loss.
    pub val_loss: f64,
    /// Training metrics.
    pub metrics: TrainingMetrics,
    /// RNG state for perfect reproducibility.
    pub rng_state: RngState,
}

/// Training metrics collected during simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training loss.
    pub train_loss: f64,
    /// Validation loss.
    pub val_loss: f64,
    /// Accuracy (if classification).
    pub accuracy: Option<f64>,
    /// Gradient L2 norm.
    pub gradient_norm: f64,
    /// Current learning rate (after scheduling).
    pub learning_rate: f64,
    /// Number of parameters updated.
    pub params_updated: usize,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            train_loss: 0.0,
            val_loss: 0.0,
            accuracy: None,
            gradient_norm: 0.0,
            learning_rate: 0.001,
            params_updated: 0,
        }
    }
}

/// Training trajectory - sequence of training states.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingTrajectory {
    /// Sequence of training states.
    pub states: Vec<TrainingState>,
}

impl TrainingTrajectory {
    /// Create new empty trajectory.
    #[must_use]
    pub fn new() -> Self {
        Self { states: Vec::new() }
    }

    /// Add a state to the trajectory.
    pub fn push(&mut self, state: TrainingState) {
        self.states.push(state);
    }

    /// Get the final training state.
    #[must_use]
    pub fn final_state(&self) -> Option<&TrainingState> {
        self.states.last()
    }

    /// Get best validation loss achieved.
    #[must_use]
    pub fn best_val_loss(&self) -> Option<f64> {
        self.states
            .iter()
            .map(|s| s.val_loss)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Check if training converged (loss stabilized).
    #[must_use]
    pub fn converged(&self, tolerance: f64) -> bool {
        if self.states.len() < 10 {
            return false;
        }
        let recent: Vec<f64> = self.states.iter().rev().take(10).map(|s| s.loss).collect();
        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance = recent.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / recent.len() as f64;
        variance.sqrt() < tolerance
    }
}

// ============================================================================
// Anomaly Detection (Jidoka)
// ============================================================================

/// Training anomaly types for Jidoka detection.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TrainingAnomaly {
    /// Loss became NaN or Infinity.
    NonFiniteLoss,
    /// Gradient norm exceeded threshold.
    GradientExplosion {
        /// Observed gradient norm.
        norm: f64,
        /// Threshold that was exceeded.
        threshold: f64,
    },
    /// Gradient norm fell below threshold.
    GradientVanishing {
        /// Observed gradient norm.
        norm: f64,
        /// Threshold that was violated.
        threshold: f64,
    },
    /// Loss spike detected (statistical outlier).
    LossSpike {
        /// Z-score of the spike.
        z_score: f64,
        /// Actual loss value.
        loss: f64,
    },
    /// Prediction confidence below threshold.
    LowConfidence {
        /// Observed confidence.
        confidence: f64,
        /// Required threshold.
        threshold: f64,
    },
}

impl std::fmt::Display for TrainingAnomaly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteLoss => write!(f, "Non-finite loss detected (NaN/Inf)"),
            Self::GradientExplosion { norm, threshold } => {
                write!(
                    f,
                    "Gradient explosion: norm={norm:.2e} > threshold={threshold:.2e}"
                )
            }
            Self::GradientVanishing { norm, threshold } => {
                write!(
                    f,
                    "Gradient vanishing: norm={norm:.2e} < threshold={threshold:.2e}"
                )
            }
            Self::LossSpike { z_score, loss } => {
                write!(f, "Loss spike: z-score={z_score:.2}, loss={loss:.4}")
            }
            Self::LowConfidence {
                confidence,
                threshold,
            } => {
                write!(
                    f,
                    "Low confidence: {confidence:.4} < threshold={threshold:.4}"
                )
            }
        }
    }
}

/// Rolling statistics for anomaly detection.
#[derive(Debug, Clone, Default)]
pub struct RollingStats {
    /// Number of observations.
    count: u64,
    /// Running mean.
    mean: f64,
    /// Running M2 for variance calculation.
    m2: f64,
    /// Window size (0 = unlimited).
    window_size: usize,
    /// Recent values for windowed stats.
    recent: Vec<f64>,
}

impl RollingStats {
    /// Create new rolling stats with optional window.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            window_size,
            recent: Vec::new(),
        }
    }

    /// Update with new observation (Welford's algorithm).
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        if self.window_size > 0 {
            self.recent.push(value);
            if self.recent.len() > self.window_size {
                self.recent.remove(0);
            }
        }
    }

    /// Get current mean.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get current variance.
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    /// Get current standard deviation.
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Compute z-score for a value.
    #[must_use]
    pub fn z_score(&self, value: f64) -> f64 {
        let std = self.std_dev();
        if std < 1e-10 {
            return 0.0;
        }
        (value - self.mean) / std
    }

    /// Reset statistics.
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
        self.recent.clear();
    }
}

/// Anomaly detector for Jidoka-style training quality gates.
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    /// Rolling statistics for loss values.
    loss_stats: RollingStats,
    /// Threshold in standard deviations for loss spikes.
    threshold_sigma: f64,
    /// Gradient explosion threshold.
    gradient_explosion_threshold: f64,
    /// Gradient vanishing threshold.
    gradient_vanishing_threshold: f64,
    /// Minimum observations before anomaly detection.
    warmup_count: u64,
    /// Number of anomalies detected.
    anomaly_count: u64,
}

impl AnomalyDetector {
    /// Create new anomaly detector with sigma threshold.
    #[must_use]
    pub fn new(threshold_sigma: f64) -> Self {
        Self {
            loss_stats: RollingStats::new(100),
            threshold_sigma,
            gradient_explosion_threshold: 1e6,
            gradient_vanishing_threshold: 1e-10,
            warmup_count: 10,
            anomaly_count: 0,
        }
    }

    /// Set gradient explosion threshold.
    #[must_use]
    pub fn with_gradient_explosion_threshold(mut self, threshold: f64) -> Self {
        self.gradient_explosion_threshold = threshold;
        self
    }

    /// Set gradient vanishing threshold.
    #[must_use]
    pub fn with_gradient_vanishing_threshold(mut self, threshold: f64) -> Self {
        self.gradient_vanishing_threshold = threshold;
        self
    }

    /// Set warmup count before anomaly detection activates.
    #[must_use]
    pub fn with_warmup(mut self, count: u64) -> Self {
        self.warmup_count = count;
        self
    }

    /// Check for training anomalies given loss and gradient norm.
    pub fn check(&mut self, loss: f64, gradient_norm: f64) -> Option<TrainingAnomaly> {
        // NaN/Inf detection (Poka-Yoke) - always active
        if !loss.is_finite() {
            self.anomaly_count += 1;
            return Some(TrainingAnomaly::NonFiniteLoss);
        }

        // Gradient explosion detection
        if gradient_norm > self.gradient_explosion_threshold {
            self.anomaly_count += 1;
            return Some(TrainingAnomaly::GradientExplosion {
                norm: gradient_norm,
                threshold: self.gradient_explosion_threshold,
            });
        }

        // Gradient vanishing detection
        if gradient_norm < self.gradient_vanishing_threshold && gradient_norm > 0.0 {
            self.anomaly_count += 1;
            return Some(TrainingAnomaly::GradientVanishing {
                norm: gradient_norm,
                threshold: self.gradient_vanishing_threshold,
            });
        }

        // Loss spike detection (statistical) - only after warmup
        self.loss_stats.update(loss);
        if self.loss_stats.count > self.warmup_count {
            let z_score = self.loss_stats.z_score(loss);
            if z_score.abs() > self.threshold_sigma {
                self.anomaly_count += 1;
                return Some(TrainingAnomaly::LossSpike { z_score, loss });
            }
        }

        None
    }

    /// Get number of anomalies detected.
    #[must_use]
    pub fn anomaly_count(&self) -> u64 {
        self.anomaly_count
    }

    /// Reset detector state.
    pub fn reset(&mut self) {
        self.loss_stats.reset();
        self.anomaly_count = 0;
    }
}

// ============================================================================
// Training Simulation
// ============================================================================

/// Simulated training event for journaling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainEvent {
    /// Epoch completed.
    Epoch(TrainingState),
    /// Anomaly detected.
    Anomaly(String),
    /// Checkpoint created.
    Checkpoint { epoch: u64 },
    /// Early stopping triggered.
    EarlyStopping { best_epoch: u64, best_val_loss: f64 },
}

/// Simulated training scenario for reproducible ML experiments.
///
/// Implements Toyota Way principles:
/// - Jidoka: Stop-on-anomaly via `AnomalyDetector`
/// - Heijunka: Load-balanced batch iteration
/// - Kaizen: Continuous improvement tracking
pub struct TrainingSimulation {
    /// Training hyperparameters.
    config: TrainingConfig,
    /// Deterministic RNG for reproducibility.
    rng: SimRng,
    /// Training event journal for replay.
    journal: EventJournal,
    /// Anomaly detector (Jidoka).
    anomaly_detector: AnomalyDetector,
    /// Current epoch.
    current_epoch: u64,
    /// Training trajectory.
    trajectory: TrainingTrajectory,
    /// Best validation loss for early stopping.
    best_val_loss: f64,
    /// Epochs without improvement counter.
    epochs_without_improvement: usize,
}

impl TrainingSimulation {
    /// Create new training simulation with deterministic seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            config: TrainingConfig::default(),
            rng: SimRng::new(seed),
            journal: EventJournal::new(true), // Record RNG state
            anomaly_detector: AnomalyDetector::new(3.0), // 3σ threshold
            current_epoch: 0,
            trajectory: TrainingTrajectory::new(),
            best_val_loss: f64::INFINITY,
            epochs_without_improvement: 0,
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(seed: u64, config: TrainingConfig) -> Self {
        Self {
            config,
            rng: SimRng::new(seed),
            journal: EventJournal::new(true), // Record RNG state
            anomaly_detector: AnomalyDetector::new(3.0),
            current_epoch: 0,
            trajectory: TrainingTrajectory::new(),
            best_val_loss: f64::INFINITY,
            epochs_without_improvement: 0,
        }
    }

    /// Set anomaly detector.
    pub fn set_anomaly_detector(&mut self, detector: AnomalyDetector) {
        self.anomaly_detector = detector;
    }

    /// Get current training configuration.
    #[must_use]
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    /// Get current trajectory.
    #[must_use]
    pub fn trajectory(&self) -> &TrainingTrajectory {
        &self.trajectory
    }

    /// Simulate a single training step with given loss and gradient norm.
    ///
    /// This is a simplified simulation - real training would compute actual
    /// forward/backward passes. This enables testing training dynamics
    /// without actual model computation.
    ///
    /// # Errors
    ///
    /// Returns error if a training anomaly is detected (Jidoka).
    pub fn step(&mut self, loss: f64, gradient_norm: f64) -> SimResult<Option<TrainingState>> {
        // Jidoka: Check for anomalies
        if let Some(anomaly) = self.anomaly_detector.check(loss, gradient_norm) {
            let event = TrainEvent::Anomaly(anomaly.to_string());
            let rng_state = self.rng.save_state();
            let _ = self.journal.append(
                SimTime::from_secs(self.current_epoch as f64),
                self.current_epoch,
                &event,
                Some(&rng_state),
            );
            return Err(SimError::jidoka(format!(
                "Training anomaly at epoch {}: {anomaly}",
                self.current_epoch
            )));
        }

        // Simulate validation loss (simplified: add noise to training loss)
        let val_loss = loss * (1.0 + 0.1 * (self.rng.gen_f64() - 0.5));

        // Create training state
        let rng_state = self.rng.save_state();
        let state = TrainingState {
            epoch: self.current_epoch,
            loss,
            val_loss,
            metrics: TrainingMetrics {
                train_loss: loss,
                val_loss,
                accuracy: None,
                gradient_norm,
                learning_rate: self.config.learning_rate,
                params_updated: 1000, // Simulated
            },
            rng_state: rng_state.clone(),
        };

        // Track best validation loss for early stopping
        if val_loss < self.best_val_loss {
            self.best_val_loss = val_loss;
            self.epochs_without_improvement = 0;
        } else {
            self.epochs_without_improvement += 1;
        }

        // Record in journal and trajectory
        let event = TrainEvent::Epoch(state.clone());
        let _ = self.journal.append(
            SimTime::from_secs(self.current_epoch as f64),
            self.current_epoch,
            &event,
            Some(&rng_state),
        );
        self.trajectory.push(state.clone());

        self.current_epoch += 1;

        // Check early stopping
        if let Some(patience) = self.config.early_stopping {
            if self.epochs_without_improvement >= patience {
                let event = TrainEvent::EarlyStopping {
                    best_epoch: self.current_epoch - patience as u64,
                    best_val_loss: self.best_val_loss,
                };
                let rng_state = self.rng.save_state();
                let _ = self.journal.append(
                    SimTime::from_secs(self.current_epoch as f64),
                    self.current_epoch,
                    &event,
                    Some(&rng_state),
                );
                return Ok(None); // Signal early stopping
            }
        }

        Ok(Some(state))
    }

    /// Simulate training for specified epochs using a loss function.
    ///
    /// The `loss_fn` takes (epoch, rng) and returns (loss, `gradient_norm`).
    ///
    /// # Errors
    ///
    /// Returns error if a training anomaly is detected.
    pub fn simulate<F>(&mut self, epochs: u64, mut loss_fn: F) -> SimResult<&TrainingTrajectory>
    where
        F: FnMut(u64, &mut SimRng) -> (f64, f64),
    {
        contract_pre_iterator!();
        for epoch in 0..epochs {
            let (loss, grad_norm) = loss_fn(epoch, &mut self.rng);
            if self.step(loss, grad_norm)?.is_none() {
                break; // Early stopping
            }
        }
        Ok(&self.trajectory)
    }

    /// Replay training from a checkpoint state.
    ///
    /// # Errors
    ///
    /// Returns error if RNG state restoration fails.
    pub fn replay_from(&mut self, checkpoint: &TrainingState) -> SimResult<()> {
        self.rng
            .restore_state(&checkpoint.rng_state)
            .map_err(|e| SimError::config(format!("Failed to restore RNG state: {e}")))?;
        self.current_epoch = checkpoint.epoch;
        Ok(())
    }

    /// Get the event journal.
    #[must_use]
    pub fn journal(&self) -> &EventJournal {
        &self.journal
    }

    /// Reset simulation state.
    pub fn reset(&mut self, seed: u64) {
        self.rng = SimRng::new(seed);
        self.journal = EventJournal::new(true);
        self.anomaly_detector.reset();
        self.current_epoch = 0;
        self.trajectory = TrainingTrajectory::new();
        self.best_val_loss = f64::INFINITY;
        self.epochs_without_improvement = 0;
    }
}
