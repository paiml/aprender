//! Calibration data collector for pruning.

use std::collections::HashMap;

use super::config::CalibrationConfig;
use super::stats::LayerActivationStats;

/// Calibration data collector for pruning.
///
/// Collects and stores per-layer activation statistics during
/// calibration forward passes.
#[derive(Debug, Clone)]
pub struct CalibrationCollector {
    /// Configuration for calibration.
    config: CalibrationConfig,
    /// Per-layer activation statistics.
    layer_stats: HashMap<String, LayerActivationStats>,
    /// Total samples processed.
    samples_processed: usize,
    /// Whether calibration is complete.
    complete: bool,
}

impl CalibrationCollector {
    /// Create a new calibration collector.
    pub fn new(config: CalibrationConfig) -> Self {
        Self { config, layer_stats: HashMap::new(), samples_processed: 0, complete: false }
    }

    /// Register a layer for calibration.
    ///
    /// # Arguments
    ///
    /// * `name` - Layer name
    /// * `input_dim` - Input feature dimension
    pub fn register_layer(&mut self, name: impl Into<String>, input_dim: usize) {
        let name = name.into();
        self.layer_stats.entry(name).or_insert_with(|| LayerActivationStats::new(input_dim));
    }

    /// Record activations for a layer.
    ///
    /// # Arguments
    ///
    /// * `layer_name` - Layer name
    /// * `activations` - Batch of activations
    pub fn record_activations(&mut self, layer_name: &str, activations: &[Vec<f32>]) {
        if self.complete {
            return;
        }

        if let Some(stats) = self.layer_stats.get_mut(layer_name) {
            stats.update(activations);
        }
    }

    /// Mark a batch as processed.
    pub fn batch_complete(&mut self, batch_size: usize) {
        self.samples_processed += batch_size;

        if self.samples_processed >= self.config.num_samples() {
            self.complete = true;
        }
    }

    /// Get activation statistics for a layer.
    pub fn get_layer_stats(&self, layer_name: &str) -> Option<&LayerActivationStats> {
        self.layer_stats.get(layer_name)
    }

    /// Get all layer names.
    pub fn layer_names(&self) -> Vec<&String> {
        self.layer_stats.keys().collect()
    }

    /// Check if calibration is complete.
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Get the number of samples processed.
    pub fn samples_processed(&self) -> usize {
        self.samples_processed
    }

    /// Get the configuration.
    pub fn config(&self) -> &CalibrationConfig {
        &self.config
    }

    /// Reset all collected statistics.
    pub fn reset(&mut self) {
        for stats in self.layer_stats.values_mut() {
            stats.reset();
        }
        self.samples_processed = 0;
        self.complete = false;
    }

    /// Get progress as a fraction (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        if self.config.num_samples() == 0 {
            return 1.0;
        }
        (self.samples_processed as f32 / self.config.num_samples() as f32).min(1.0)
    }
}
