//! Pixel-Level GUI Coverage Visualization (Advanced Feature A)
//!
//! Grid-based coverage tracking that renders heatmaps showing which
//! screen regions have been exercised by tests. Identifies untested
//! visual regions between UI elements.

mod config;
mod falsification;
mod heatmap;
mod metrics;
mod parallel;
mod terminal;
mod tracker;
mod wasm_demo;

pub use config::{
    ConfigValidationError, OutputConfig, PerformanceConfig, PixelCoverageConfig, ThresholdConfig,
    VerificationConfig,
};
pub use falsification::{
    ComparisonOperator, FalsifiabilityGate, FalsifiableHypothesis, FalsifiableHypothesisBuilder,
    FalsificationCondition, FalsificationLayer, GateResult,
};
pub use heatmap::{
    BitmapFont, ColorPalette, HeatmapRenderer, PngHeatmap, Rgb, StatsPanel, TerminalHeatmap,
};
pub use metrics::{
    CieDe2000Metric, DeltaEClassification, DeltaEResult, Lab, PerceptualHash, PhashAlgorithm,
    PixelVerificationResult, PixelVerificationSuite, PsnrMetric, PsnrQuality, PsnrResult,
    SsimMetric, SsimResult,
};
pub use parallel::{
    BatchProcessor, DeltaEBatchResult, Downscaler, HashCache, ParallelContext, SsimBatchResult,
};
pub use terminal::{
    ansi, ConfidenceInterval, CoverageHypothesis, GapRegion, OutputMode, RichTerminalHeatmap,
    ScoreBar,
};
pub use tracker::{
    CombinedCoverageReport, CoverageCell, GridConfig, LineCoverageReport, PixelCoverageReport,
    PixelCoverageTracker, Point as PixelPoint, Region as PixelRegion,
};
pub use wasm_demo::{
    wilson_confidence_interval, ConfigError as WasmDemoConfigError,
    CoverageStats as WasmCoverageStats, DemoGapRegion, DemoPalette, GapSeverity as DemoGapSeverity,
    GpuPixelBuffer, PcgRng, WasmDemoConfig, WasmPixelDemo,
};

/// Coverage threshold presets
pub mod thresholds {
    /// Minimum acceptable coverage (60%)
    pub const MINIMUM: f32 = 0.60;
    /// Standard coverage target (80%)
    pub const STANDARD: f32 = 0.80;
    /// High coverage target (90%)
    pub const HIGH: f32 = 0.90;
    /// Complete coverage (100%)
    pub const COMPLETE: f32 = 1.0;
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::assertions_on_constants
)]
mod tests {
    use super::tracker::{Point, Region};
    use super::*;

    // =========================================================================
    // H₀-PIX-01: PixelCoverageTracker creation
    // =========================================================================

    #[test]
    fn h0_pix_01_tracker_creation() {
        let tracker = PixelCoverageTracker::new(1920, 1080, 64, 36);
        assert_eq!(tracker.resolution(), (1920, 1080));
        assert_eq!(tracker.grid_size(), (64, 36));
    }

    #[test]
    fn h0_pix_02_tracker_builder() {
        let tracker = PixelCoverageTracker::builder()
            .resolution(1280, 720)
            .grid_size(32, 18)
            .threshold(0.75)
            .build();

        assert_eq!(tracker.resolution(), (1280, 720));
        assert_eq!(tracker.grid_size(), (32, 18));
        assert!((tracker.threshold() - 0.75).abs() < f32::EPSILON);
    }

    // =========================================================================
    // H₀-PIX-03: Coverage recording
    // =========================================================================

    #[test]
    fn h0_pix_03_record_interaction() {
        let mut tracker = PixelCoverageTracker::new(100, 100, 10, 10);

        // Record interaction at (50, 50) - should affect cell (5, 5)
        tracker.record_interaction(Point::new(50, 50));

        let report = tracker.generate_report();
        assert!(report.overall_coverage > 0.0);
    }

    #[test]
    fn h0_pix_04_record_region() {
        let mut tracker = PixelCoverageTracker::new(100, 100, 10, 10);

        // Record a region covering multiple cells
        tracker.record_region(Region::new(0, 0, 50, 50));

        let report = tracker.generate_report();
        // Should cover 25% of the grid (5x5 cells out of 10x10)
        assert!(report.overall_coverage >= 0.20);
    }

    // =========================================================================
    // H₀-PIX-05: Coverage report
    // =========================================================================

    #[test]
    fn h0_pix_05_report_empty_tracker() {
        let tracker = PixelCoverageTracker::new(100, 100, 10, 10);
        let report = tracker.generate_report();

        assert_eq!(report.overall_coverage, 0.0);
        assert!(!report.meets_threshold);
    }

    #[test]
    fn h0_pix_06_report_full_coverage() {
        let mut tracker = PixelCoverageTracker::new(100, 100, 10, 10);

        // Cover entire screen
        tracker.record_region(Region::new(0, 0, 100, 100));

        let report = tracker.generate_report();
        assert_eq!(report.overall_coverage, 1.0);
        assert!(report.meets_threshold);
    }

    // =========================================================================
    // H₀-PIX-07: Uncovered regions
    // =========================================================================

    #[test]
    fn h0_pix_07_find_uncovered_regions() {
        let mut tracker = PixelCoverageTracker::new(100, 100, 10, 10);

        // Cover only top-left quadrant
        tracker.record_region(Region::new(0, 0, 50, 50));

        let uncovered = tracker.uncovered_regions();
        assert!(!uncovered.is_empty());
    }

    // =========================================================================
    // H₀-PIX-08: Heatmap rendering
    // =========================================================================

    #[test]
    fn h0_pix_08_terminal_heatmap() {
        let mut tracker = PixelCoverageTracker::new(100, 100, 10, 10);
        tracker.record_region(Region::new(0, 0, 50, 50));

        let heatmap = tracker.terminal_heatmap();
        let rendered = heatmap.render();

        assert!(!rendered.is_empty());
        assert!(rendered.contains('█') || rendered.contains('░'));
    }

    // =========================================================================
    // H₀-PIX-09: Point and Region types
    // =========================================================================

    #[test]
    fn h0_pix_09_point_creation() {
        let point = Point::new(100, 200);
        assert_eq!(point.x, 100);
        assert_eq!(point.y, 200);
    }

    #[test]
    fn h0_pix_10_region_creation() {
        let region = Region::new(10, 20, 30, 40);
        assert_eq!(region.x, 10);
        assert_eq!(region.y, 20);
        assert_eq!(region.width, 30);
        assert_eq!(region.height, 40);
    }

    #[test]
    fn h0_pix_11_region_contains() {
        let region = Region::new(10, 10, 50, 50);

        assert!(region.contains(Point::new(20, 20)));
        assert!(region.contains(Point::new(10, 10))); // Edge case
        assert!(!region.contains(Point::new(5, 5))); // Outside
        assert!(!region.contains(Point::new(70, 70))); // Outside
    }

    // =========================================================================
    // H₀-PIX-12: Grid configuration
    // =========================================================================

    #[test]
    fn h0_pix_12_grid_cell_size() {
        let tracker = PixelCoverageTracker::new(640, 480, 64, 48);
        let config = tracker.grid_config();

        assert_eq!(config.cell_width(), 10); // 640 / 64
        assert_eq!(config.cell_height(), 10); // 480 / 48
    }

    // =========================================================================
    // H₀-PIX-13: Threshold presets
    // =========================================================================

    #[test]
    fn h0_pix_13_threshold_presets() {
        assert!(thresholds::MINIMUM < thresholds::STANDARD);
        assert!(thresholds::STANDARD < thresholds::HIGH);
        assert!(thresholds::HIGH < thresholds::COMPLETE);
    }
}
