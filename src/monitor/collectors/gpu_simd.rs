//! SIMD-accelerated GPU metrics aggregator.
//!
//! This module provides a SIMD-optimized wrapper for GPU metrics that works
//! with the existing GPU collectors (NVIDIA, AMD, Apple) while adding:
//!
//! - SIMD-accelerated history storage using SimdRingBuffer
//! - Vectorized multi-GPU aggregation
//! - Cross-GPU statistics computation
//!
//! ## Performance Targets (Falsifiable)
//!
//! - History updates: < 5μs per GPU
//! - Statistics computation: < 10μs for 8 GPUs
//!
//! ## Design
//!
//! The actual GPU metric collection is delegated to hardware-specific collectors
//! (NVML for NVIDIA, ROCm for AMD, Metal for Apple). This module focuses on
//! SIMD-accelerating the post-collection processing and history management.

use crate::monitor::simd::ring_buffer::SimdRingBuffer;
use crate::monitor::simd::{kernels, SimdStats};

/// Maximum supported GPUs.
pub const MAX_GPUS: usize = 16;

/// SIMD-optimized GPU metrics container.
///
/// Stores metrics for multiple GPUs in SoA layout for SIMD-friendly access.
#[repr(C, align(64))]
#[derive(Debug)]
pub struct GpuMetricsSoA {
    /// GPU utilization per device (0-100).
    pub gpu_util: Vec<f64>,
    /// Memory utilization per device (0-100).
    pub mem_util: Vec<f64>,
    /// Temperature per device (Celsius).
    pub temperature: Vec<f64>,
    /// Power draw per device (milliwatts).
    pub power_mw: Vec<u64>,
    /// Power limit per device (milliwatts).
    pub power_limit_mw: Vec<u64>,
    /// Memory used per device (bytes).
    pub mem_used: Vec<u64>,
    /// Memory total per device (bytes).
    pub mem_total: Vec<u64>,
    /// Number of GPUs.
    pub gpu_count: usize,
}

impl GpuMetricsSoA {
    /// Creates a new GPU metrics container for the specified GPU count.
    #[must_use]
    pub fn new(gpu_count: usize) -> Self {
        let count = gpu_count.min(MAX_GPUS);
        let aligned_count = count.div_ceil(8) * 8;

        Self {
            gpu_util: vec![0.0; aligned_count],
            mem_util: vec![0.0; aligned_count],
            temperature: vec![0.0; aligned_count],
            power_mw: vec![0; aligned_count],
            power_limit_mw: vec![0; aligned_count],
            mem_used: vec![0; aligned_count],
            mem_total: vec![0; aligned_count],
            gpu_count: count,
        }
    }

    /// Sets metrics for a specific GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn set_gpu(
        &mut self,
        index: usize,
        gpu_util: f64,
        mem_util: f64,
        temperature: f64,
        power_mw: u64,
        power_limit_mw: u64,
        mem_used: u64,
        mem_total: u64,
    ) {
        if index >= self.gpu_util.len() {
            return;
        }

        self.gpu_util[index] = gpu_util;
        self.mem_util[index] = mem_util;
        self.temperature[index] = temperature;
        self.power_mw[index] = power_mw;
        self.power_limit_mw[index] = power_limit_mw;
        self.mem_used[index] = mem_used;
        self.mem_total[index] = mem_total;
    }

    /// Returns average GPU utilization using SIMD.
    #[must_use]
    pub fn avg_gpu_util(&self) -> f64 {
        if self.gpu_count == 0 {
            return 0.0;
        }
        kernels::simd_mean(&self.gpu_util[..self.gpu_count])
    }

    /// Returns average memory utilization using SIMD.
    #[must_use]
    pub fn avg_mem_util(&self) -> f64 {
        if self.gpu_count == 0 {
            return 0.0;
        }
        kernels::simd_mean(&self.mem_util[..self.gpu_count])
    }

    /// Returns maximum temperature using SIMD.
    #[must_use]
    pub fn max_temperature(&self) -> f64 {
        if self.gpu_count == 0 {
            return 0.0;
        }
        kernels::simd_max(&self.temperature[..self.gpu_count])
    }

    /// Returns total power draw in watts.
    #[must_use]
    pub fn total_power_watts(&self) -> f64 {
        let power_f64: Vec<f64> =
            self.power_mw[..self.gpu_count].iter().map(|&p| p as f64 / 1000.0).collect();
        kernels::simd_sum(&power_f64)
    }

    /// Returns total memory used in bytes.
    #[must_use]
    pub fn total_mem_used(&self) -> u64 {
        self.mem_used[..self.gpu_count].iter().sum()
    }

    /// Returns total memory capacity in bytes.
    #[must_use]
    pub fn total_mem_total(&self) -> u64 {
        self.mem_total[..self.gpu_count].iter().sum()
    }
}

impl Default for GpuMetricsSoA {
    fn default() -> Self {
        Self::new(0)
    }
}

/// SIMD-accelerated GPU history manager.
///
/// Provides efficient history storage and statistics for GPU metrics.
#[derive(Debug)]
pub struct SimdGpuHistory {
    /// GPU utilization history per GPU.
    gpu_util_history: Vec<SimdRingBuffer>,
    /// Memory utilization history per GPU.
    mem_util_history: Vec<SimdRingBuffer>,
    /// Temperature history per GPU.
    temp_history: Vec<SimdRingBuffer>,
    /// Power history per GPU (normalized 0-1).
    power_history: Vec<SimdRingBuffer>,
    /// Aggregate GPU utilization history (average across GPUs).
    aggregate_gpu_history: SimdRingBuffer,
    /// Aggregate memory utilization history.
    aggregate_mem_history: SimdRingBuffer,
    /// Number of GPUs.
    gpu_count: usize,
}

impl SimdGpuHistory {
    /// Creates a new GPU history manager for the specified GPU count.
    #[must_use]
    pub fn new(gpu_count: usize) -> Self {
        let count = gpu_count.min(MAX_GPUS);
        let history_size = 300; // 5 minutes at 1Hz

        let mut gpu_util_history = Vec::with_capacity(count);
        let mut mem_util_history = Vec::with_capacity(count);
        let mut temp_history = Vec::with_capacity(count);
        let mut power_history = Vec::with_capacity(count);

        for _ in 0..count {
            gpu_util_history.push(SimdRingBuffer::new(history_size));
            mem_util_history.push(SimdRingBuffer::new(history_size));
            temp_history.push(SimdRingBuffer::new(history_size));
            power_history.push(SimdRingBuffer::new(history_size));
        }

        Self {
            gpu_util_history,
            mem_util_history,
            temp_history,
            power_history,
            aggregate_gpu_history: SimdRingBuffer::new(history_size),
            aggregate_mem_history: SimdRingBuffer::new(history_size),
            gpu_count: count,
        }
    }

    /// Updates history with current metrics.
    pub fn update(&mut self, metrics: &GpuMetricsSoA) {
        for i in 0..self.gpu_count.min(metrics.gpu_count) {
            // Normalize to 0-1 range for history
            self.gpu_util_history[i].push(metrics.gpu_util[i] / 100.0);
            self.mem_util_history[i].push(metrics.mem_util[i] / 100.0);

            // Temperature normalized to 0-1 (assuming max 100°C)
            self.temp_history[i].push(metrics.temperature[i] / 100.0);

            // Power normalized to power limit
            let power_norm = if metrics.power_limit_mw[i] > 0 {
                metrics.power_mw[i] as f64 / metrics.power_limit_mw[i] as f64
            } else {
                0.0
            };
            self.power_history[i].push(power_norm.min(1.0));
        }

        // Update aggregate history
        self.aggregate_gpu_history.push(metrics.avg_gpu_util() / 100.0);
        self.aggregate_mem_history.push(metrics.avg_mem_util() / 100.0);
    }

    /// Returns GPU utilization history for a specific GPU.
    #[must_use]
    pub fn gpu_util_history(&self, gpu_idx: usize) -> Option<&SimdRingBuffer> {
        self.gpu_util_history.get(gpu_idx)
    }

    /// Returns memory utilization history for a specific GPU.
    #[must_use]
    pub fn mem_util_history(&self, gpu_idx: usize) -> Option<&SimdRingBuffer> {
        self.mem_util_history.get(gpu_idx)
    }

    /// Returns temperature history for a specific GPU.
    #[must_use]
    pub fn temp_history(&self, gpu_idx: usize) -> Option<&SimdRingBuffer> {
        self.temp_history.get(gpu_idx)
    }

    /// Returns power history for a specific GPU.
    #[must_use]
    pub fn power_history(&self, gpu_idx: usize) -> Option<&SimdRingBuffer> {
        self.power_history.get(gpu_idx)
    }

    /// Returns aggregate GPU utilization history.
    #[must_use]
    pub fn aggregate_gpu_history(&self) -> &SimdRingBuffer {
        &self.aggregate_gpu_history
    }

    /// Returns aggregate memory utilization history.
    #[must_use]
    pub fn aggregate_mem_history(&self) -> &SimdRingBuffer {
        &self.aggregate_mem_history
    }

    /// Returns GPU utilization statistics for a specific GPU.
    #[must_use]
    pub fn gpu_util_stats(&self, gpu_idx: usize) -> Option<&SimdStats> {
        self.gpu_util_history
            .get(gpu_idx)
            .map(super::super::simd::ring_buffer::SimdRingBuffer::statistics)
    }

    /// Returns aggregate GPU utilization statistics.
    #[must_use]
    pub fn aggregate_gpu_stats(&self) -> &SimdStats {
        self.aggregate_gpu_history.statistics()
    }

    /// Returns the number of GPUs tracked.
    #[must_use]
    pub fn gpu_count(&self) -> usize {
        self.gpu_count
    }

    /// Clears all history.
    pub fn clear(&mut self) {
        for h in &mut self.gpu_util_history {
            h.clear();
        }
        for h in &mut self.mem_util_history {
            h.clear();
        }
        for h in &mut self.temp_history {
            h.clear();
        }
        for h in &mut self.power_history {
            h.clear();
        }
        self.aggregate_gpu_history.clear();
        self.aggregate_mem_history.clear();
    }
}

impl Default for SimdGpuHistory {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_metrics_soa_new() {
        let metrics = GpuMetricsSoA::new(4);
        assert_eq!(metrics.gpu_count, 4);
        assert!(metrics.gpu_util.len() >= 4);
    }

    #[test]
    fn test_gpu_metrics_soa_set_gpu() {
        let mut metrics = GpuMetricsSoA::new(4);
        metrics.set_gpu(
            0, 75.0,      // gpu_util
            50.0,      // mem_util
            65.0,      // temperature
            150_000,   // power_mw (150W)
            300_000,   // power_limit_mw (300W)
            4_000_000, // mem_used
            8_000_000, // mem_total
        );

        assert!((metrics.gpu_util[0] - 75.0).abs() < 0.01);
        assert!((metrics.mem_util[0] - 50.0).abs() < 0.01);
        assert!((metrics.temperature[0] - 65.0).abs() < 0.01);
        assert_eq!(metrics.power_mw[0], 150_000);
    }

    #[test]
    fn test_gpu_metrics_soa_avg_gpu_util() {
        let mut metrics = GpuMetricsSoA::new(4);
        metrics.gpu_util[0] = 50.0;
        metrics.gpu_util[1] = 60.0;
        metrics.gpu_util[2] = 70.0;
        metrics.gpu_util[3] = 80.0;
        metrics.gpu_count = 4;

        let avg = metrics.avg_gpu_util();
        assert!((avg - 65.0).abs() < 0.1);
    }

    #[test]
    fn test_gpu_metrics_soa_max_temperature() {
        let mut metrics = GpuMetricsSoA::new(4);
        metrics.temperature[0] = 55.0;
        metrics.temperature[1] = 65.0;
        metrics.temperature[2] = 75.0;
        metrics.temperature[3] = 60.0;
        metrics.gpu_count = 4;

        let max = metrics.max_temperature();
        assert!((max - 75.0).abs() < 0.1);
    }

    #[test]
    fn test_gpu_metrics_soa_total_power() {
        let mut metrics = GpuMetricsSoA::new(4);
        metrics.power_mw[0] = 150_000; // 150W
        metrics.power_mw[1] = 200_000; // 200W
        metrics.gpu_count = 2;

        let total = metrics.total_power_watts();
        assert!((total - 350.0).abs() < 0.1);
    }

    #[test]
    fn test_simd_gpu_history_new() {
        let history = SimdGpuHistory::new(4);
        assert_eq!(history.gpu_count(), 4);
    }

    #[test]
    fn test_simd_gpu_history_update() {
        let mut history = SimdGpuHistory::new(2);
        let mut metrics = GpuMetricsSoA::new(2);

        metrics.set_gpu(0, 75.0, 50.0, 65.0, 150_000, 300_000, 4_000_000, 8_000_000);
        metrics.set_gpu(1, 80.0, 60.0, 70.0, 200_000, 350_000, 6_000_000, 12_000_000);

        history.update(&metrics);

        // Check that history was updated
        assert_eq!(history.gpu_util_history(0).expect("value should be present").len(), 1);
        assert_eq!(history.gpu_util_history(1).expect("value should be present").len(), 1);

        // Check normalized values
        let gpu0_latest = history.gpu_util_history(0).expect("operation should succeed").latest();
        assert!((gpu0_latest.expect("value should be present") - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_simd_gpu_history_aggregate() {
        let mut history = SimdGpuHistory::new(2);
        let mut metrics = GpuMetricsSoA::new(2);

        metrics.set_gpu(0, 50.0, 40.0, 60.0, 100_000, 300_000, 2_000_000, 8_000_000);
        metrics.set_gpu(1, 70.0, 60.0, 70.0, 150_000, 350_000, 4_000_000, 12_000_000);

        history.update(&metrics);

        // Aggregate should be average: (50+70)/2 = 60%
        let agg = history.aggregate_gpu_history().latest().expect("operation should succeed");
        assert!((agg - 0.60).abs() < 0.01);
    }

    #[test]
    fn test_simd_gpu_history_clear() {
        let mut history = SimdGpuHistory::new(2);
        let mut metrics = GpuMetricsSoA::new(2);

        metrics.set_gpu(0, 50.0, 40.0, 60.0, 100_000, 300_000, 2_000_000, 8_000_000);
        history.update(&metrics);

        assert!(!history.aggregate_gpu_history().is_empty());

        history.clear();

        assert!(history.aggregate_gpu_history().is_empty());
    }

    #[test]
    fn test_simd_gpu_history_stats() {
        let mut history = SimdGpuHistory::new(1);
        let mut metrics = GpuMetricsSoA::new(1);

        // Push multiple samples
        for i in 0..10 {
            metrics.gpu_util[0] = 50.0 + (f64::from(i) * 5.0);
            metrics.gpu_count = 1;
            history.update(&metrics);
        }

        let stats = history.gpu_util_stats(0).expect("operation should succeed");
        assert!(stats.count >= 10);
    }

    #[test]
    fn test_alignment() {
        assert_eq!(std::mem::align_of::<GpuMetricsSoA>(), 64);
    }
}
