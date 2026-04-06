//! PTQ Calibrator implementation
//!
//! The main `Calibrator` struct for collecting statistics and computing
//! quantization parameters.

use crate::Tensor;

use super::helpers::rand_simple;
use super::types::{CalibrationMethod, CalibrationResult};

/// PTQ Calibrator for collecting statistics and computing quantization parameters
#[derive(Clone, Debug)]
pub struct Calibrator {
    /// Calibration method
    method: CalibrationMethod,
    /// Whether quantization is symmetric
    symmetric: bool,
    /// Number of bits for quantization
    bits: usize,
    /// Running minimum (for moving average)
    running_min: Option<f32>,
    /// Running maximum (for moving average)
    running_max: Option<f32>,
    /// Collected samples (for percentile)
    samples: Vec<f32>,
    /// Maximum samples to collect (for percentile)
    max_samples: usize,
    /// Number of batches observed
    num_batches: usize,
}

impl Calibrator {
    /// Create new calibrator with min-max method
    pub fn min_max(bits: usize, symmetric: bool) -> Self {
        Self {
            method: CalibrationMethod::MinMax,
            symmetric,
            bits,
            running_min: None,
            running_max: None,
            samples: Vec::new(),
            max_samples: 0,
            num_batches: 0,
        }
    }

    /// Create new calibrator with percentile method
    ///
    /// # Arguments
    /// * `bits` - Number of quantization bits
    /// * `symmetric` - Whether to use symmetric quantization
    /// * `lower` - Lower percentile (e.g., 0.01 for 0.01%)
    /// * `upper` - Upper percentile (e.g., 99.99 for 99.99%)
    /// * `max_samples` - Maximum number of samples to collect
    pub fn percentile(
        bits: usize,
        symmetric: bool,
        lower: f32,
        upper: f32,
        max_samples: usize,
    ) -> Self {
        Self {
            method: CalibrationMethod::Percentile { lower, upper },
            symmetric,
            bits,
            running_min: None,
            running_max: None,
            samples: Vec::with_capacity(max_samples.min(10000)),
            max_samples,
            num_batches: 0,
        }
    }

    /// Create new calibrator with moving average method
    pub fn moving_average(bits: usize, symmetric: bool, momentum: f32) -> Self {
        Self {
            method: CalibrationMethod::MovingAverage { momentum },
            symmetric,
            bits,
            running_min: None,
            running_max: None,
            samples: Vec::new(),
            max_samples: 0,
            num_batches: 0,
        }
    }

    /// Observe a batch of data for calibration
    pub fn observe(&mut self, data: &[f32]) {
        if data.is_empty() {
            return;
        }

        match &self.method {
            CalibrationMethod::MinMax => {
                self.observe_min_max(data);
            }
            CalibrationMethod::Percentile { .. } => {
                self.observe_percentile(data);
            }
            CalibrationMethod::MovingAverage { momentum } => {
                let momentum = *momentum;
                self.observe_moving_average(data, momentum);
            }
        }

        self.num_batches += 1;
    }

    /// Observe a tensor for calibration
    pub fn observe_tensor(&mut self, tensor: &Tensor) {
        if let Some(slice) = tensor.data().as_slice() {
            self.observe(slice);
        }
    }

    /// Observe multiple tensors
    pub fn observe_tensors(&mut self, tensors: &[&Tensor]) {
        for tensor in tensors {
            self.observe_tensor(tensor);
        }
    }

    /// Compute calibration result
    pub fn compute(&self) -> CalibrationResult {
        let (observed_min, observed_max) = match &self.method {
            CalibrationMethod::MinMax | CalibrationMethod::MovingAverage { .. } => {
                (self.running_min.unwrap_or(0.0), self.running_max.unwrap_or(0.0))
            }
            CalibrationMethod::Percentile { lower, upper } => {
                self.compute_percentile_bounds(*lower, *upper)
            }
        };

        let (scale, zero_point) = self.compute_scale_zero_point(observed_min, observed_max);

        CalibrationResult {
            scale,
            zero_point,
            observed_min,
            observed_max,
            method: self.method.clone(),
        }
    }

    /// Get number of batches observed
    pub fn num_batches(&self) -> usize {
        self.num_batches
    }

    /// Get calibration method
    pub fn method(&self) -> &CalibrationMethod {
        &self.method
    }

    /// Check if any data has been observed
    pub fn has_data(&self) -> bool {
        self.num_batches > 0
    }

    /// Reset calibration state
    pub fn reset(&mut self) {
        self.running_min = None;
        self.running_max = None;
        self.samples.clear();
        self.num_batches = 0;
    }

    // Internal methods

    fn observe_min_max(&mut self, data: &[f32]) {
        let batch_min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let batch_max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        self.running_min = Some(self.running_min.map_or(batch_min, |m| m.min(batch_min)));
        self.running_max = Some(self.running_max.map_or(batch_max, |m| m.max(batch_max)));
    }

    fn observe_percentile(&mut self, data: &[f32]) {
        // Collect samples (with reservoir sampling if needed)
        if self.samples.len() < self.max_samples {
            let remaining = self.max_samples - self.samples.len();
            self.samples.extend(data.iter().take(remaining).copied());
        } else {
            // Reservoir sampling for samples beyond max_samples
            let total_seen = self.num_batches * data.len() + data.len();
            for (i, &val) in data.iter().enumerate() {
                let j = rand_simple(total_seen + i);
                if j < self.max_samples {
                    self.samples[j] = val;
                }
            }
        }

        // Also track min/max for fallback
        self.observe_min_max(data);
    }

    fn observe_moving_average(&mut self, data: &[f32], momentum: f32) {
        let batch_min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let batch_max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        self.running_min = Some(
            self.running_min.map_or(batch_min, |m| m * (1.0 - momentum) + batch_min * momentum),
        );
        self.running_max = Some(
            self.running_max.map_or(batch_max, |m| m * (1.0 - momentum) + batch_max * momentum),
        );
    }

    fn compute_percentile_bounds(&self, lower: f32, upper: f32) -> (f32, f32) {
        if self.samples.is_empty() {
            return (self.running_min.unwrap_or(0.0), self.running_max.unwrap_or(0.0));
        }

        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let lower_idx = ((lower / 100.0) * n as f32) as usize;
        let upper_idx = ((upper / 100.0) * n as f32).min((n - 1) as f32) as usize;

        (sorted[lower_idx], sorted[upper_idx])
    }

    fn compute_scale_zero_point(&self, min_val: f32, max_val: f32) -> (f32, i32) {
        let qmax = (1 << (self.bits - 1)) - 1;
        let qmin = if self.symmetric { -qmax } else { 0 };
        let qmax_full = if self.symmetric { qmax } else { (1 << self.bits) - 1 };

        if self.symmetric {
            // Symmetric: scale from max absolute value
            let max_abs = min_val.abs().max(max_val.abs());
            let scale = if max_abs < 1e-10 { 1e-10 } else { max_abs / qmax as f32 };
            (scale, 0)
        } else {
            // Asymmetric: scale from range
            let range = max_val - min_val;
            let scale = if range < 1e-10 { 1e-10 } else { range / (qmax_full - qmin) as f32 };
            let zero_point = (qmin as f32 - min_val / scale).round() as i32;
            let zero_point = zero_point.clamp(qmin, qmax_full);
            (scale, zero_point)
        }
    }
}
