//! Cross-Backend Regression Detector (PMAT-031)
//!
//! Detect performance regressions when switching between compute backends.
//!
//! # Features
//!
//! - Compare efficiency across backends (Scalar, SSE2, AVX2, CUDA, Metal)
//! - Detect size thresholds where performance cliffs occur
//! - Recommend optimal backend for given workload size
//! - Measure GPU transfer overhead vs compute benefit
//!
//! # Falsification Criteria (F1231-F1240)
//!
//! See `tests/backend_regression_f1231.rs` for falsification tests.

mod analysis;
mod detector;
mod types;

pub use analysis::{BackendSummary, TransferAnalysis};
pub use types::{
    Backend, BackendComparison, BackendMeasurement, BackendRecommendation, SizeCliff, WorkloadType,
};

use std::collections::HashSet;

/// Backend regression detector
#[derive(Debug, Clone)]
pub struct BackendRegressionDetector {
    /// Measurements collected
    measurements: Vec<BackendMeasurement>,
    /// Regression threshold (default 10%)
    threshold_percent: f64,
    /// Cliff detection threshold (default 10%)
    cliff_threshold_percent: f64,
    /// Available backends
    available_backends: Vec<Backend>,
}

impl Default for BackendRegressionDetector {
    fn default() -> Self {
        Self {
            measurements: Vec::new(),
            threshold_percent: 10.0,
            cliff_threshold_percent: 10.0,
            available_backends: vec![Backend::Scalar, Backend::Sse2, Backend::Avx2],
        }
    }
}

impl BackendRegressionDetector {
    /// Create new detector
    pub fn new() -> Self {
        Self::default()
    }

    /// Set regression threshold
    pub fn with_threshold(mut self, percent: f64) -> Self {
        self.threshold_percent = percent;
        self
    }

    /// Set cliff detection threshold
    pub fn with_cliff_threshold(mut self, percent: f64) -> Self {
        self.cliff_threshold_percent = percent;
        self
    }

    /// Set available backends
    pub fn with_backends(mut self, backends: Vec<Backend>) -> Self {
        self.available_backends = backends;
        self
    }

    /// Add a measurement
    pub fn add_measurement(&mut self, measurement: BackendMeasurement) {
        self.measurements.push(measurement);
    }

    /// Add measurement from values
    pub fn add(
        &mut self,
        backend: Backend,
        workload: WorkloadType,
        size: usize,
        latency_us: f64,
        throughput: f64,
        efficiency: f64,
    ) {
        self.add_measurement(
            BackendMeasurement::new(backend, workload, size, latency_us, throughput)
                .with_efficiency(efficiency),
        );
    }

    /// Get measurement count
    pub fn measurement_count(&self) -> usize {
        self.measurements.len()
    }

    /// Access measurements slice
    pub(crate) fn measurements(&self) -> &[BackendMeasurement] {
        &self.measurements
    }

    /// Access threshold percent
    pub(crate) fn threshold_percent(&self) -> f64 {
        self.threshold_percent
    }

    /// Access cliff threshold percent
    pub(crate) fn cliff_threshold_percent(&self) -> f64 {
        self.cliff_threshold_percent
    }

    /// Collect unique values from measurements via an extractor.
    pub(crate) fn unique<T: Eq + std::hash::Hash + Copy>(
        &self,
        f: impl Fn(&BackendMeasurement) -> T,
    ) -> Vec<T> {
        self.measurements
            .iter()
            .map(f)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    }

    /// Collect unique values from measurements matching a workload.
    pub(crate) fn unique_for<T: Eq + std::hash::Hash + Copy>(
        &self,
        workload: WorkloadType,
        f: impl Fn(&BackendMeasurement) -> T,
    ) -> Vec<T> {
        self.measurements
            .iter()
            .filter(|m| m.workload == workload)
            .map(f)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    }

    /// Find measurement
    pub(crate) fn find_measurement(
        &self,
        backend: Backend,
        workload: WorkloadType,
        size: usize,
    ) -> Option<&BackendMeasurement> {
        self.measurements
            .iter()
            .find(|m| m.backend == backend && m.workload == workload && m.size == size)
    }

    /// Check if backend is available
    pub fn is_backend_available(&self, backend: Backend) -> bool {
        self.available_backends.contains(&backend)
    }

    /// Get available backends
    pub fn available_backends(&self) -> &[Backend] {
        &self.available_backends
    }

    /// Clear all measurements
    pub fn clear(&mut self) {
        self.measurements.clear();
    }
}

#[cfg(test)]
mod tests;
