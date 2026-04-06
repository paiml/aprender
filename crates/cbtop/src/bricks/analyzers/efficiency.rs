//! Efficiency analyzer brick (Layer 2)
//!
//! Calculates compute efficiency metrics based on theoretical vs actual throughput.
//! Uses roofline model principles to determine bottleneck type.

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use std::any::Any;

/// Efficiency classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EfficiencyClass {
    /// Excellent efficiency (>90%)
    Excellent,
    /// Good efficiency (70-90%)
    Good,
    /// Fair efficiency (50-70%)
    Fair,
    /// Poor efficiency (<50%)
    Poor,
    /// Unknown (no data)
    #[default]
    Unknown,
}

impl EfficiencyClass {
    /// Get efficiency class from percentage
    pub fn from_percent(pct: f64) -> Self {
        if pct >= 90.0 {
            Self::Excellent
        } else if pct >= 70.0 {
            Self::Good
        } else if pct >= 50.0 {
            Self::Fair
        } else if pct > 0.0 {
            Self::Poor
        } else {
            Self::Unknown
        }
    }

    /// Display name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Excellent => "Excellent",
            Self::Good => "Good",
            Self::Fair => "Fair",
            Self::Poor => "Poor",
            Self::Unknown => "Unknown",
        }
    }

    /// Color hint for UI (green=good, red=poor)
    pub fn color_hint(&self) -> (f32, f32, f32) {
        match self {
            Self::Excellent => (0.3, 1.0, 0.5), // Green
            Self::Good => (0.5, 1.0, 0.3),      // Yellow-green
            Self::Fair => (1.0, 0.8, 0.2),      // Yellow
            Self::Poor => (1.0, 0.3, 0.2),      // Red
            Self::Unknown => (0.5, 0.5, 0.5),   // Gray
        }
    }
}

/// Bottleneck type detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BottleneckType {
    /// Compute-bound (FLOPS limited)
    ComputeBound,
    /// Memory-bound (bandwidth limited)
    MemoryBound,
    /// Latency-bound (serialization/dependency)
    LatencyBound,
    /// Thermal-bound (throttling)
    ThermalBound,
    /// PCIe-bound (transfer overhead)
    PcieBound,
    /// Balanced (no clear bottleneck)
    #[default]
    Balanced,
}

impl BottleneckType {
    /// Display name
    pub fn name(&self) -> &'static str {
        match self {
            Self::ComputeBound => "Compute-Bound",
            Self::MemoryBound => "Memory-Bound",
            Self::LatencyBound => "Latency-Bound",
            Self::ThermalBound => "Thermal-Bound",
            Self::PcieBound => "PCIe-Bound",
            Self::Balanced => "Balanced",
        }
    }

    /// Recommendation for improvement
    pub fn recommendation(&self) -> &'static str {
        match self {
            Self::ComputeBound => "Increase parallelism or use higher FLOPS hardware",
            Self::MemoryBound => "Improve data locality, use tiling, or increase bandwidth",
            Self::LatencyBound => "Reduce dependencies, increase batch size",
            Self::ThermalBound => "Improve cooling or reduce power target",
            Self::PcieBound => "Batch transfers, use pinned memory, or compute on device",
            Self::Balanced => "System is well balanced - no single bottleneck",
        }
    }
}

/// Efficiency metrics
#[derive(Debug, Clone, Default)]
pub struct EfficiencyMetrics {
    /// Compute efficiency (actual FLOPS / peak FLOPS * 100)
    pub compute_efficiency: f64,
    /// Memory efficiency (actual bandwidth / peak bandwidth * 100)
    pub memory_efficiency: f64,
    /// Overall efficiency (weighted combination)
    pub overall_efficiency: f64,
    /// Arithmetic intensity (FLOPS per byte)
    pub arithmetic_intensity: f64,
    /// Primary bottleneck
    pub bottleneck: BottleneckType,
    /// Efficiency classification
    pub classification: EfficiencyClass,
}

impl EfficiencyMetrics {
    /// Create metrics from raw measurements
    pub fn calculate(
        actual_flops: f64,
        peak_flops: f64,
        actual_bandwidth: f64,
        peak_bandwidth: f64,
        operations: u64,
        bytes_transferred: u64,
    ) -> Self {
        let compute_efficiency =
            if peak_flops > 0.0 { (actual_flops / peak_flops * 100.0).min(100.0) } else { 0.0 };

        let memory_efficiency = if peak_bandwidth > 0.0 {
            (actual_bandwidth / peak_bandwidth * 100.0).min(100.0)
        } else {
            0.0
        };

        let arithmetic_intensity =
            if bytes_transferred > 0 { operations as f64 / bytes_transferred as f64 } else { 0.0 };

        // Determine bottleneck based on relative efficiencies
        let bottleneck = if compute_efficiency < 30.0 && memory_efficiency < 30.0 {
            BottleneckType::LatencyBound
        } else if compute_efficiency > memory_efficiency + 20.0 {
            BottleneckType::MemoryBound
        } else if memory_efficiency > compute_efficiency + 20.0 {
            BottleneckType::ComputeBound
        } else {
            BottleneckType::Balanced
        };

        // Overall efficiency is weighted average
        let overall_efficiency = (compute_efficiency + memory_efficiency) / 2.0;
        let classification = EfficiencyClass::from_percent(overall_efficiency);

        Self {
            compute_efficiency,
            memory_efficiency,
            overall_efficiency,
            arithmetic_intensity,
            bottleneck,
            classification,
        }
    }
}

/// Efficiency analyzer brick
pub struct EfficiencyAnalyzerBrick {
    /// Current efficiency metrics
    pub metrics: EfficiencyMetrics,
    /// Peak FLOPS for the device (GFLOPS)
    pub peak_flops: f64,
    /// Peak memory bandwidth (GB/s)
    pub peak_bandwidth: f64,
    /// History of overall efficiency samples
    pub efficiency_history: Vec<f64>,
    /// History length limit
    pub history_limit: usize,
}

impl EfficiencyAnalyzerBrick {
    /// Create a new efficiency analyzer with device specs
    pub fn new(peak_flops: f64, peak_bandwidth: f64) -> Self {
        Self {
            metrics: EfficiencyMetrics::default(),
            peak_flops,
            peak_bandwidth,
            efficiency_history: Vec::new(),
            history_limit: 120,
        }
    }

    /// Create with reasonable defaults for a mid-range GPU
    pub fn with_defaults() -> Self {
        // Reasonable defaults: ~10 TFLOPS, ~500 GB/s
        Self::new(10000.0, 500.0)
    }

    /// Update metrics with new measurements
    pub fn update(
        &mut self,
        actual_flops: f64,
        actual_bandwidth: f64,
        operations: u64,
        bytes_transferred: u64,
    ) {
        self.metrics = EfficiencyMetrics::calculate(
            actual_flops,
            self.peak_flops,
            actual_bandwidth,
            self.peak_bandwidth,
            operations,
            bytes_transferred,
        );

        // Track history
        self.efficiency_history.push(self.metrics.overall_efficiency);
        if self.efficiency_history.len() > self.history_limit {
            self.efficiency_history.remove(0);
        }
    }

    /// Set thermal throttling detected
    pub fn set_thermal_throttling(&mut self, is_throttling: bool) {
        if is_throttling {
            self.metrics.bottleneck = BottleneckType::ThermalBound;
        }
    }

    /// Set PCIe bottleneck detected
    pub fn set_pcie_bottleneck(&mut self, is_bottleneck: bool) {
        if is_bottleneck && self.metrics.bottleneck == BottleneckType::Balanced {
            self.metrics.bottleneck = BottleneckType::PcieBound;
        }
    }

    /// Get average efficiency over history
    pub fn average_efficiency(&self) -> f64 {
        if self.efficiency_history.is_empty() {
            return 0.0;
        }
        self.efficiency_history.iter().sum::<f64>() / self.efficiency_history.len() as f64
    }

    /// Get efficiency trend (positive = improving, negative = degrading)
    pub fn efficiency_trend(&self) -> f64 {
        if self.efficiency_history.len() < 10 {
            return 0.0;
        }
        let recent: f64 = self.efficiency_history.iter().rev().take(5).sum::<f64>() / 5.0;
        let older: f64 = self.efficiency_history.iter().rev().skip(5).take(5).sum::<f64>() / 5.0;
        recent - older
    }
}

impl Default for EfficiencyAnalyzerBrick {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl Brick for EfficiencyAnalyzerBrick {
    fn brick_name(&self) -> &'static str {
        "efficiency_analyzer"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![BrickAssertion::max_latency_ms(4)]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::FRAME_60FPS
    }

    fn verify(&self) -> BrickVerification {
        let mut v = BrickVerification::new();
        for assertion in self.assertions() {
            v.check(&assertion);
        }
        v
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_efficiency_analyzer_brick_name() {
        let analyzer = EfficiencyAnalyzerBrick::with_defaults();
        assert_eq!(analyzer.brick_name(), "efficiency_analyzer");
    }

    #[test]
    fn test_efficiency_class_from_percent() {
        assert_eq!(EfficiencyClass::from_percent(95.0), EfficiencyClass::Excellent);
        assert_eq!(EfficiencyClass::from_percent(80.0), EfficiencyClass::Good);
        assert_eq!(EfficiencyClass::from_percent(60.0), EfficiencyClass::Fair);
        assert_eq!(EfficiencyClass::from_percent(30.0), EfficiencyClass::Poor);
        assert_eq!(EfficiencyClass::from_percent(0.0), EfficiencyClass::Unknown);
    }

    #[test]
    fn test_efficiency_metrics_calculate() {
        let metrics = EfficiencyMetrics::calculate(
            5000.0,    // actual GFLOPS
            10000.0,   // peak GFLOPS
            250.0,     // actual GB/s
            500.0,     // peak GB/s
            1_000_000, // operations
            100_000,   // bytes
        );

        assert!((metrics.compute_efficiency - 50.0).abs() < 0.01);
        assert!((metrics.memory_efficiency - 50.0).abs() < 0.01);
        assert!((metrics.overall_efficiency - 50.0).abs() < 0.01);
        assert!((metrics.arithmetic_intensity - 10.0).abs() < 0.01);
        assert_eq!(metrics.bottleneck, BottleneckType::Balanced);
        assert_eq!(metrics.classification, EfficiencyClass::Fair);
    }

    #[test]
    fn test_memory_bound_detection() {
        let metrics = EfficiencyMetrics::calculate(
            9000.0, // high compute utilization
            10000.0, 100.0, // low memory utilization
            500.0, 1_000_000, 100_000,
        );

        assert_eq!(metrics.bottleneck, BottleneckType::MemoryBound);
    }

    #[test]
    fn test_compute_bound_detection() {
        let metrics = EfficiencyMetrics::calculate(
            1000.0, // low compute utilization
            10000.0, 450.0, // high memory utilization
            500.0, 1_000_000, 100_000,
        );

        assert_eq!(metrics.bottleneck, BottleneckType::ComputeBound);
    }

    #[test]
    fn test_latency_bound_detection() {
        let metrics = EfficiencyMetrics::calculate(
            100.0, // very low compute
            10000.0, 50.0, // very low memory
            500.0, 1_000_000, 100_000,
        );

        assert_eq!(metrics.bottleneck, BottleneckType::LatencyBound);
    }

    #[test]
    fn test_efficiency_history() {
        let mut analyzer = EfficiencyAnalyzerBrick::new(10000.0, 500.0);

        for i in 0..10 {
            analyzer.update((5000 + i * 100) as f64, 250.0, 1_000_000, 100_000);
        }

        assert_eq!(analyzer.efficiency_history.len(), 10);
        assert!(analyzer.average_efficiency() > 0.0);
    }

    #[test]
    fn test_efficiency_trend() {
        let mut analyzer = EfficiencyAnalyzerBrick::new(10000.0, 500.0);

        // Add improving efficiency samples
        for i in 0..15 {
            analyzer.update((2000 + i * 500) as f64, 250.0, 1_000_000, 100_000);
        }

        let trend = analyzer.efficiency_trend();
        assert!(trend > 0.0, "Trend should be positive for improving efficiency");
    }

    #[test]
    fn test_history_limit() {
        let mut analyzer = EfficiencyAnalyzerBrick::new(10000.0, 500.0);
        analyzer.history_limit = 10;

        for _ in 0..20 {
            analyzer.update(5000.0, 250.0, 1_000_000, 100_000);
        }

        assert_eq!(analyzer.efficiency_history.len(), 10);
    }

    #[test]
    fn test_bottleneck_recommendations() {
        assert!(!BottleneckType::ComputeBound.recommendation().is_empty());
        assert!(!BottleneckType::MemoryBound.recommendation().is_empty());
        assert!(!BottleneckType::LatencyBound.recommendation().is_empty());
        assert!(!BottleneckType::ThermalBound.recommendation().is_empty());
        assert!(!BottleneckType::PcieBound.recommendation().is_empty());
        assert!(!BottleneckType::Balanced.recommendation().is_empty());
    }

    #[test]
    fn test_set_thermal_throttling() {
        let mut analyzer = EfficiencyAnalyzerBrick::with_defaults();
        analyzer.update(5000.0, 250.0, 1_000_000, 100_000);

        analyzer.set_thermal_throttling(true);
        assert_eq!(analyzer.metrics.bottleneck, BottleneckType::ThermalBound);
    }

    #[test]
    fn test_efficiency_class_color_hints() {
        let (r, g, _b) = EfficiencyClass::Excellent.color_hint();
        assert!(g > r, "Excellent should be greenish");

        let (r, g, _) = EfficiencyClass::Poor.color_hint();
        assert!(r > g, "Poor should be reddish");
    }
}
