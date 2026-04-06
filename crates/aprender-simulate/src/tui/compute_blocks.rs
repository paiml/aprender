//! `ComputeBlock` integration for simular TUI
//!
//! Provides SIMD-optimized visualization widgets from presentar-terminal
//! for simulation metrics. Integrates with probar's `ComputeBlockAssertion`
//! for testability.
//!
//! # Architecture (SIMULAR-CB-001)
//!
//! ```text
//! OrbitApp State
//!     │
//!     ├─► EnergySparkline (SparklineBlock)
//!     │   └─► Normalized energy history graph
//!     │
//!     ├─► MomentumSparkline (SparklineBlock)
//!     │   └─► Normalized angular momentum history
//!     │
//!     └─► FrameBudgetTrend (LoadTrendBlock)
//!         └─► Heijunka utilization over time
//! ```
//!
//! # Toyota Way Integration
//!
//! - **Jidoka**: `SparklineBlocks` show conservation violations
//! - **Mieruka**: Visual management of simulation health
//! - **Heijunka**: Frame budget visualization

use presentar_terminal::{ComputeBlock, LoadTrendBlock, SimdInstructionSet, SparklineBlock};

/// Maximum history length for sparklines (SPEC-024 compliant)
const SPARKLINE_HISTORY: usize = 60;

/// Default render width for sparklines
const SPARKLINE_WIDTH: usize = 40;

/// Energy conservation sparkline using presentar-terminal's SIMD-optimized `SparklineBlock`
pub struct EnergySparkline {
    block: SparklineBlock,
    initial_energy: Option<f64>,
}

impl EnergySparkline {
    /// Create a new energy sparkline
    #[must_use]
    #[provable_contracts_macros::contract("simular-sparkline-v1", equation = "new")]
    pub fn new() -> Self {
        Self {
            block: SparklineBlock::new(SPARKLINE_HISTORY),
            initial_energy: None,
        }
    }

    /// Push a new energy value
    pub fn push(&mut self, energy: f64) {
        // Store initial energy for normalization
        if self.initial_energy.is_none() {
            self.initial_energy = Some(energy);
        }

        // Compute relative energy drift (should stay near 0 for good conservation)
        let initial = self.initial_energy.unwrap_or(energy);
        #[allow(clippy::cast_possible_truncation)]
        let drift = if initial.abs() > f64::EPSILON {
            ((energy - initial) / initial.abs() * 1e6) as f32 // Parts per million
        } else {
            0.0
        };

        self.block.compute(&drift);
    }

    /// Get the rendered sparkline characters
    #[must_use]
    pub fn render(&self) -> Vec<char> {
        self.block.render(SPARKLINE_WIDTH)
    }

    /// Get the min/max range for display
    #[must_use]
    pub fn range(&self) -> (f32, f32) {
        let history = self.block.history();
        if history.is_empty() {
            return (0.0, 0.0);
        }
        let min = history.iter().copied().fold(f32::INFINITY, f32::min);
        let max = history.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (min, max)
    }

    /// Get current value
    #[must_use]
    pub fn current(&self) -> Option<f32> {
        self.block.history().last().copied()
    }

    /// Get history length
    #[must_use]
    pub fn len(&self) -> usize {
        self.block.history().len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.block.history().is_empty()
    }

    /// Get SIMD instruction set being used
    #[must_use]
    pub fn simd_instruction_set(&self) -> SimdInstructionSet {
        self.block.simd_instruction_set()
    }

    /// Reset the sparkline
    pub fn reset(&mut self) {
        self.block = SparklineBlock::new(SPARKLINE_HISTORY);
        self.initial_energy = None;
    }
}

impl Default for EnergySparkline {
    fn default() -> Self {
        Self::new()
    }
}

/// Angular momentum conservation sparkline
pub struct MomentumSparkline {
    block: SparklineBlock,
    initial_momentum: Option<f64>,
}

impl MomentumSparkline {
    /// Create a new momentum sparkline
    #[must_use]
    pub fn new() -> Self {
        Self {
            block: SparklineBlock::new(SPARKLINE_HISTORY),
            initial_momentum: None,
        }
    }

    /// Push a new angular momentum value
    pub fn push(&mut self, momentum: f64) {
        if self.initial_momentum.is_none() {
            self.initial_momentum = Some(momentum);
        }

        let initial = self.initial_momentum.unwrap_or(momentum);
        #[allow(clippy::cast_possible_truncation)]
        let drift = if initial.abs() > f64::EPSILON {
            ((momentum - initial) / initial.abs() * 1e6) as f32
        } else {
            0.0
        };

        self.block.compute(&drift);
    }

    /// Get the rendered sparkline characters
    #[must_use]
    pub fn render(&self) -> Vec<char> {
        self.block.render(SPARKLINE_WIDTH)
    }

    /// Get the min/max range
    #[must_use]
    pub fn range(&self) -> (f32, f32) {
        let history = self.block.history();
        if history.is_empty() {
            return (0.0, 0.0);
        }
        let min = history.iter().copied().fold(f32::INFINITY, f32::min);
        let max = history.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (min, max)
    }

    /// Get current value
    #[must_use]
    pub fn current(&self) -> Option<f32> {
        self.block.history().last().copied()
    }

    /// Get history length
    #[must_use]
    pub fn len(&self) -> usize {
        self.block.history().len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.block.history().is_empty()
    }

    /// Get SIMD instruction set
    #[must_use]
    pub fn simd_instruction_set(&self) -> SimdInstructionSet {
        self.block.simd_instruction_set()
    }

    /// Reset the sparkline
    pub fn reset(&mut self) {
        self.block = SparklineBlock::new(SPARKLINE_HISTORY);
        self.initial_momentum = None;
    }
}

impl Default for MomentumSparkline {
    fn default() -> Self {
        Self::new()
    }
}

/// Re-export `TrendDirection` from presentar-terminal
pub use presentar_terminal::compute_block::TrendDirection;

/// Frame budget utilization trend (Heijunka visualization)
pub struct FrameBudgetTrend {
    block: LoadTrendBlock,
    /// History of utilization values for sparkline rendering
    history: Vec<f32>,
    /// Sparkline for visual rendering
    sparkline: SparklineBlock,
}

impl FrameBudgetTrend {
    /// Create a new frame budget trend
    #[must_use]
    pub fn new() -> Self {
        Self {
            block: LoadTrendBlock::new(SPARKLINE_HISTORY),
            history: Vec::with_capacity(SPARKLINE_HISTORY),
            sparkline: SparklineBlock::new(SPARKLINE_HISTORY),
        }
    }

    /// Push a new utilization value (0.0 - 1.0+)
    pub fn push(&mut self, utilization: f64) {
        // LoadTrendBlock expects 0-100 scale
        #[allow(clippy::cast_possible_truncation)]
        let pct = (utilization * 100.0).clamp(0.0, 200.0) as f32;
        self.block.compute(&pct);
        self.sparkline.compute(&pct);

        // Track history for average calculation
        if self.history.len() >= SPARKLINE_HISTORY {
            self.history.remove(0);
        }
        self.history.push(pct);
    }

    /// Get the rendered trend characters (using sparkline)
    #[must_use]
    pub fn render(&self) -> Vec<char> {
        self.sparkline.render(SPARKLINE_WIDTH)
    }

    /// Get the average utilization
    #[must_use]
    pub fn average(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history.iter().sum::<f32>() / self.history.len() as f32
    }

    /// Get trend direction
    #[must_use]
    pub fn trend(&self) -> TrendDirection {
        self.block.trend()
    }

    /// Get history length
    #[must_use]
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Get SIMD instruction set
    #[must_use]
    pub fn simd_instruction_set(&self) -> SimdInstructionSet {
        self.sparkline.simd_instruction_set()
    }

    /// Reset the trend
    pub fn reset(&mut self) {
        self.block = LoadTrendBlock::new(SPARKLINE_HISTORY);
        self.sparkline = SparklineBlock::new(SPARKLINE_HISTORY);
        self.history.clear();
    }
}

impl Default for FrameBudgetTrend {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined simulation metrics using `ComputeBlocks`
pub struct SimulationMetrics {
    /// Energy conservation sparkline
    pub energy: EnergySparkline,
    /// Angular momentum conservation sparkline
    pub momentum: MomentumSparkline,
    /// Frame budget utilization trend
    pub frame_budget: FrameBudgetTrend,
}

impl SimulationMetrics {
    /// Create new simulation metrics
    #[must_use]
    pub fn new() -> Self {
        Self {
            energy: EnergySparkline::new(),
            momentum: MomentumSparkline::new(),
            frame_budget: FrameBudgetTrend::new(),
        }
    }

    /// Update all metrics from simulation state
    pub fn update(&mut self, energy: f64, momentum: f64, frame_utilization: f64) {
        self.energy.push(energy);
        self.momentum.push(momentum);
        self.frame_budget.push(frame_utilization);
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.energy.reset();
        self.momentum.reset();
        self.frame_budget.reset();
    }

    /// Get the SIMD instruction set being used
    #[must_use]
    pub fn simd_instruction_set(&self) -> SimdInstructionSet {
        // All blocks use the same detection
        self.energy.simd_instruction_set()
    }
}

impl Default for SimulationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ====== EnergySparkline tests ======

    #[test]
    fn test_energy_sparkline_creation() {
        let sparkline = EnergySparkline::new();
        assert!(sparkline.is_empty());
        assert_eq!(sparkline.len(), 0);
    }

    #[test]
    fn test_energy_sparkline_push() {
        let mut sparkline = EnergySparkline::new();
        sparkline.push(-1e30);
        assert!(!sparkline.is_empty());
        assert_eq!(sparkline.len(), 1);
    }

    #[test]
    fn test_energy_sparkline_drift_calculation() {
        let mut sparkline = EnergySparkline::new();
        sparkline.push(-1e30);
        sparkline.push(-1e30 * 1.000001); // 1 ppm drift

        let current = sparkline.current();
        assert!(current.is_some());
    }

    #[test]
    fn test_energy_sparkline_render() {
        let mut sparkline = EnergySparkline::new();
        for i in 0..10 {
            sparkline.push(-1e30 + (i as f64) * 1e20);
        }
        let chars = sparkline.render();
        assert!(!chars.is_empty());
    }

    #[test]
    fn test_energy_sparkline_range_empty() {
        let sparkline = EnergySparkline::new();
        let (min, max) = sparkline.range();
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
    }

    #[test]
    fn test_energy_sparkline_range_with_values() {
        let mut sparkline = EnergySparkline::new();
        sparkline.push(-1e30);
        sparkline.push(-1e30 * 1.001); // Some drift
        sparkline.push(-1e30 * 0.999); // Some negative drift

        let (min, max) = sparkline.range();
        assert!(min <= max);
    }

    #[test]
    fn test_energy_sparkline_current_empty() {
        let sparkline = EnergySparkline::new();
        assert!(sparkline.current().is_none());
    }

    #[test]
    fn test_energy_sparkline_simd() {
        let sparkline = EnergySparkline::new();
        let simd = sparkline.simd_instruction_set();
        assert!(simd.vector_width() >= 1);
    }

    #[test]
    fn test_energy_sparkline_reset() {
        let mut sparkline = EnergySparkline::new();
        sparkline.push(-1e30);
        sparkline.push(-1e30 * 1.001);
        assert_eq!(sparkline.len(), 2);

        sparkline.reset();
        assert!(sparkline.is_empty());
        assert!(sparkline.current().is_none());
    }

    #[test]
    fn test_energy_sparkline_zero_initial() {
        // Test the else branch when initial energy is zero/epsilon
        let mut sparkline = EnergySparkline::new();
        sparkline.push(0.0);
        sparkline.push(1.0);

        let current = sparkline.current();
        assert!(current.is_some());
        // Should be 0.0 because initial was 0.0 (near epsilon)
        assert_eq!(current.unwrap(), 0.0);
    }

    // ====== MomentumSparkline tests ======

    #[test]
    fn test_momentum_sparkline_creation() {
        let sparkline = MomentumSparkline::new();
        assert!(sparkline.is_empty());
        assert_eq!(sparkline.len(), 0);
    }

    #[test]
    fn test_momentum_sparkline_push() {
        let mut sparkline = MomentumSparkline::new();
        sparkline.push(1e40);
        assert_eq!(sparkline.len(), 1);
        assert!(!sparkline.is_empty());
    }

    #[test]
    fn test_momentum_sparkline_render() {
        let mut sparkline = MomentumSparkline::new();
        for i in 0..10 {
            sparkline.push(1e40 + (i as f64) * 1e30);
        }
        let chars = sparkline.render();
        assert!(!chars.is_empty());
    }

    #[test]
    fn test_momentum_sparkline_range_empty() {
        let sparkline = MomentumSparkline::new();
        let (min, max) = sparkline.range();
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
    }

    #[test]
    fn test_momentum_sparkline_range_with_values() {
        let mut sparkline = MomentumSparkline::new();
        sparkline.push(1e40);
        sparkline.push(1e40 * 1.001);
        sparkline.push(1e40 * 0.999);

        let (min, max) = sparkline.range();
        assert!(min <= max);
    }

    #[test]
    fn test_momentum_sparkline_current_empty() {
        let sparkline = MomentumSparkline::new();
        assert!(sparkline.current().is_none());
    }

    #[test]
    fn test_momentum_sparkline_current_with_values() {
        let mut sparkline = MomentumSparkline::new();
        sparkline.push(1e40);
        sparkline.push(1e40 * 1.001);

        let current = sparkline.current();
        assert!(current.is_some());
    }

    #[test]
    fn test_momentum_sparkline_simd() {
        let sparkline = MomentumSparkline::new();
        let simd = sparkline.simd_instruction_set();
        assert!(simd.vector_width() >= 1);
    }

    #[test]
    fn test_momentum_sparkline_reset() {
        let mut sparkline = MomentumSparkline::new();
        sparkline.push(1e40);
        sparkline.push(1e40 * 1.001);
        assert_eq!(sparkline.len(), 2);

        sparkline.reset();
        assert!(sparkline.is_empty());
        assert!(sparkline.current().is_none());
    }

    #[test]
    fn test_momentum_sparkline_zero_initial() {
        let mut sparkline = MomentumSparkline::new();
        sparkline.push(0.0);
        sparkline.push(100.0);

        let current = sparkline.current();
        assert!(current.is_some());
        assert_eq!(current.unwrap(), 0.0);
    }

    // ====== FrameBudgetTrend tests ======

    #[test]
    fn test_frame_budget_trend_creation() {
        let trend = FrameBudgetTrend::new();
        assert!(trend.is_empty());
        assert_eq!(trend.len(), 0);
    }

    #[test]
    fn test_frame_budget_trend_push() {
        let mut trend = FrameBudgetTrend::new();
        trend.push(0.5); // 50% utilization
        assert_eq!(trend.len(), 1);
        assert!(!trend.is_empty());
    }

    #[test]
    fn test_frame_budget_trend_render() {
        let mut trend = FrameBudgetTrend::new();
        for i in 0..10 {
            trend.push(0.3 + (i as f64) * 0.05);
        }
        let chars = trend.render();
        assert!(!chars.is_empty());
    }

    #[test]
    fn test_frame_budget_trend_average_empty() {
        let trend = FrameBudgetTrend::new();
        assert_eq!(trend.average(), 0.0);
    }

    #[test]
    fn test_frame_budget_trend_average_with_values() {
        let mut trend = FrameBudgetTrend::new();
        trend.push(0.5); // 50%
        trend.push(0.7); // 70%
        trend.push(0.8); // 80%

        let avg = trend.average();
        // Average of 50, 70, 80 = 66.67
        assert!((avg - 66.67).abs() < 1.0);
    }

    #[test]
    fn test_frame_budget_trend_direction() {
        let mut trend = FrameBudgetTrend::new();
        // Push increasing values to get an upward trend
        for i in 0..10 {
            trend.push(0.3 + (i as f64) * 0.05);
        }
        let _direction = trend.trend();
        // Just verify it returns a direction
    }

    #[test]
    fn test_frame_budget_trend_simd() {
        let trend = FrameBudgetTrend::new();
        let simd = trend.simd_instruction_set();
        assert!(simd.vector_width() >= 1);
    }

    #[test]
    fn test_frame_budget_trend_reset() {
        let mut trend = FrameBudgetTrend::new();
        trend.push(0.5);
        trend.push(0.7);
        assert_eq!(trend.len(), 2);

        trend.reset();
        assert!(trend.is_empty());
        assert_eq!(trend.average(), 0.0);
    }

    #[test]
    fn test_frame_budget_trend_history_overflow() {
        let mut trend = FrameBudgetTrend::new();
        // Push more than SPARKLINE_HISTORY values
        for i in 0..70 {
            trend.push((i as f64) * 0.01);
        }
        // Should be capped at SPARKLINE_HISTORY (60)
        assert_eq!(trend.len(), SPARKLINE_HISTORY);
    }

    #[test]
    fn test_frame_budget_trend_clamp() {
        let mut trend = FrameBudgetTrend::new();
        // Test clamping at 200%
        trend.push(3.0); // Should clamp to 200%
        assert_eq!(trend.len(), 1);
    }

    // ====== SimulationMetrics tests ======

    #[test]
    fn test_simulation_metrics_creation() {
        let metrics = SimulationMetrics::new();
        assert!(metrics.energy.is_empty());
        assert!(metrics.momentum.is_empty());
        assert!(metrics.frame_budget.is_empty());
    }

    #[test]
    fn test_simulation_metrics_update() {
        let mut metrics = SimulationMetrics::new();
        metrics.update(-1e30, 1e40, 0.5);

        assert_eq!(metrics.energy.len(), 1);
        assert_eq!(metrics.momentum.len(), 1);
        assert_eq!(metrics.frame_budget.len(), 1);
    }

    #[test]
    fn test_simulation_metrics_reset() {
        let mut metrics = SimulationMetrics::new();
        metrics.update(-1e30, 1e40, 0.5);
        metrics.update(-1e30 * 1.001, 1e40 * 1.001, 0.6);
        metrics.reset();

        assert!(metrics.energy.is_empty());
        assert!(metrics.momentum.is_empty());
        assert!(metrics.frame_budget.is_empty());
    }

    #[test]
    fn test_simulation_metrics_simd() {
        let metrics = SimulationMetrics::new();
        let simd = metrics.simd_instruction_set();
        // Should detect something (at least Scalar)
        assert!(simd.vector_width() >= 1);
    }

    #[test]
    fn test_simulation_metrics_multiple_updates() {
        let mut metrics = SimulationMetrics::new();

        for i in 0..20 {
            let energy = -1e30 * (1.0 + (i as f64) * 0.0001);
            let momentum = 1e40 * (1.0 + (i as f64) * 0.0001);
            let util = 0.3 + (i as f64) * 0.02;
            metrics.update(energy, momentum, util);
        }

        assert_eq!(metrics.energy.len(), 20);
        assert_eq!(metrics.momentum.len(), 20);
        assert_eq!(metrics.frame_budget.len(), 20);

        // Verify ranges are populated
        let (e_min, e_max) = metrics.energy.range();
        assert!(e_min <= e_max);

        let (m_min, m_max) = metrics.momentum.range();
        assert!(m_min <= m_max);

        let avg = metrics.frame_budget.average();
        assert!(avg > 0.0);
    }

    // ====== Default trait tests ======

    #[test]
    fn test_defaults() {
        let energy = EnergySparkline::default();
        assert!(energy.is_empty());

        let momentum = MomentumSparkline::default();
        assert!(momentum.is_empty());

        let frame = FrameBudgetTrend::default();
        assert!(frame.is_empty());

        let metrics = SimulationMetrics::default();
        assert!(metrics.energy.is_empty());
    }

    // ====== TrendDirection re-export test ======

    #[test]
    fn test_trend_direction_variants() {
        // Verify the re-export works
        let _up = TrendDirection::Up;
        let _down = TrendDirection::Down;
        let _flat = TrendDirection::Flat;
    }
}
