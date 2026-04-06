//! Calibration data loader implementation.

use super::config::CalibrationDataConfig;
use super::iter::CalibrationDataIter;
use crate::train::Batch;
use crate::Tensor;

/// Calibration data loader for pruning.
///
/// Provides an iterator over calibration batches for collecting
/// activation statistics during pruning.
#[derive(Debug, Clone)]
pub struct CalibrationDataLoader {
    /// Configuration.
    config: CalibrationDataConfig,
    /// Pre-loaded data (if available).
    data: Option<Vec<Batch>>,
    /// Current position in iteration.
    pub(crate) position: usize,
}

impl CalibrationDataLoader {
    /// Create a new calibration data loader.
    pub fn new(config: CalibrationDataConfig) -> Self {
        Self { config, data: None, position: 0 }
    }

    /// Create a data loader with pre-loaded synthetic data for testing.
    pub fn with_synthetic_data(config: CalibrationDataConfig) -> Self {
        let mut loader = Self::new(config);
        loader.generate_synthetic_data();
        loader
    }

    /// Generate synthetic calibration data for testing.
    fn generate_synthetic_data(&mut self) {
        use rand::prelude::*;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(self.config.seed());
        let mut batches = Vec::with_capacity(self.config.num_batches());

        let mut samples_remaining = self.config.num_samples();

        while samples_remaining > 0 {
            let batch_size = samples_remaining.min(self.config.batch_size());
            samples_remaining -= batch_size;

            // Generate random input data
            let input_size = batch_size * self.config.sequence_length();
            let inputs: Vec<f32> = (0..input_size).map(|_| rng.random::<f32>()).collect();
            let targets: Vec<f32> = (0..batch_size).map(|_| rng.random::<f32>()).collect();

            batches.push(Batch::new(
                Tensor::from_vec(inputs, false),
                Tensor::from_vec(targets, false),
            ));
        }

        self.data = Some(batches);
    }

    /// Load data from the configured source.
    ///
    /// This is a placeholder for actual dataset loading.
    /// In production, this would load from C4, WikiText, etc.
    pub fn load(&mut self) -> Result<(), String> {
        if self.data.is_some() {
            return Ok(());
        }

        // For now, generate synthetic data
        // Real implementation would load from dataset
        self.generate_synthetic_data();
        Ok(())
    }

    /// Get the configuration.
    pub fn config(&self) -> &CalibrationDataConfig {
        &self.config
    }

    /// Check if data is loaded.
    pub fn is_loaded(&self) -> bool {
        self.data.is_some()
    }

    /// Get the number of batches.
    pub fn num_batches(&self) -> usize {
        self.data.as_ref().map_or(0, Vec::len)
    }

    /// Reset the iterator position.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Get a batch by index.
    pub fn get_batch(&self, index: usize) -> Option<&Batch> {
        self.data.as_ref().and_then(|d| d.get(index))
    }

    /// Create an iterator over batches.
    pub fn iter(&self) -> CalibrationDataIter<'_> {
        CalibrationDataIter::new(self)
    }
}

impl<'a> IntoIterator for &'a CalibrationDataLoader {
    type Item = &'a Batch;
    type IntoIter = CalibrationDataIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
