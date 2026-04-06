//! Pixel Coverage Tracker Implementation
//!
//! Tracks which grid cells have been exercised during testing.

use serde::{Deserialize, Serialize};

/// A point in screen coordinates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Point {
    /// X coordinate (pixels from left)
    pub x: u32,
    /// Y coordinate (pixels from top)
    pub y: u32,
}

impl Point {
    /// Create a new point
    #[must_use]
    pub const fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }
}

/// A rectangular region in screen coordinates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Region {
    /// X coordinate of top-left corner
    pub x: u32,
    /// Y coordinate of top-left corner
    pub y: u32,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
}

impl Region {
    /// Create a new region
    #[must_use]
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Check if a point is within this region
    #[must_use]
    pub fn contains(&self, point: Point) -> bool {
        point.x >= self.x
            && point.x < self.x + self.width
            && point.y >= self.y
            && point.y < self.y + self.height
    }

    /// Get the area of this region
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// Grid configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GridConfig {
    /// Screen width in pixels
    pub screen_width: u32,
    /// Screen height in pixels
    pub screen_height: u32,
    /// Number of grid columns
    pub grid_cols: u32,
    /// Number of grid rows
    pub grid_rows: u32,
}

impl GridConfig {
    /// Calculate cell width in pixels
    #[must_use]
    pub fn cell_width(&self) -> u32 {
        self.screen_width / self.grid_cols
    }

    /// Calculate cell height in pixels
    #[must_use]
    pub fn cell_height(&self) -> u32 {
        self.screen_height / self.grid_rows
    }

    /// Convert screen coordinates to grid cell
    #[must_use]
    pub fn point_to_cell(&self, point: Point) -> (u32, u32) {
        let col = (point.x / self.cell_width()).min(self.grid_cols - 1);
        let row = (point.y / self.cell_height()).min(self.grid_rows - 1);
        (col, row)
    }

    /// Convert grid cell to screen region
    #[must_use]
    pub fn cell_to_region(&self, col: u32, row: u32) -> Region {
        Region::new(
            col * self.cell_width(),
            row * self.cell_height(),
            self.cell_width(),
            self.cell_height(),
        )
    }
}

/// A single coverage cell in the grid
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoverageCell {
    /// Number of times this cell was interacted with
    pub hit_count: u64,
    /// Coverage value (0.0 - 1.0)
    pub coverage: f32,
}

impl CoverageCell {
    /// Check if cell is covered
    #[must_use]
    pub fn is_covered(&self) -> bool {
        self.hit_count > 0
    }
}

/// Pixel coverage tracker for grid-based UI coverage
#[derive(Debug, Clone)]
pub struct PixelCoverageTracker {
    config: GridConfig,
    cells: Vec<Vec<CoverageCell>>,
    threshold: f32,
    total_interactions: u64,
}

impl PixelCoverageTracker {
    /// Create a new tracker with given resolution and grid size
    #[must_use]
    pub fn new(width: u32, height: u32, grid_cols: u32, grid_rows: u32) -> Self {
        let config = GridConfig {
            screen_width: width,
            screen_height: height,
            grid_cols,
            grid_rows,
        };

        let cells = (0..grid_rows)
            .map(|_| (0..grid_cols).map(|_| CoverageCell::default()).collect())
            .collect();

        Self {
            config,
            cells,
            threshold: 0.8,
            total_interactions: 0,
        }
    }

    /// Create a builder for more complex configuration
    #[must_use]
    pub fn builder() -> PixelCoverageTrackerBuilder {
        PixelCoverageTrackerBuilder::default()
    }

    /// Get screen resolution
    #[must_use]
    pub fn resolution(&self) -> (u32, u32) {
        (self.config.screen_width, self.config.screen_height)
    }

    /// Get grid size
    #[must_use]
    pub fn grid_size(&self) -> (u32, u32) {
        (self.config.grid_cols, self.config.grid_rows)
    }

    /// Get coverage threshold
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Get grid configuration
    #[must_use]
    pub fn grid_config(&self) -> &GridConfig {
        &self.config
    }

    /// Record an interaction at a point
    pub fn record_interaction(&mut self, point: Point) {
        let (col, row) = self.config.point_to_cell(point);
        if let Some(row_cells) = self.cells.get_mut(row as usize) {
            if let Some(cell) = row_cells.get_mut(col as usize) {
                cell.hit_count += 1;
                cell.coverage = 1.0;
                self.total_interactions += 1;
            }
        }
    }

    /// Record coverage for a region
    pub fn record_region(&mut self, region: Region) {
        let start_col = region.x / self.config.cell_width();
        let start_row = region.y / self.config.cell_height();
        let end_col =
            ((region.x + region.width) / self.config.cell_width()).min(self.config.grid_cols - 1);
        let end_row =
            ((region.y + region.height) / self.config.cell_height()).min(self.config.grid_rows - 1);

        for row in start_row..=end_row {
            for col in start_col..=end_col {
                if let Some(row_cells) = self.cells.get_mut(row as usize) {
                    if let Some(cell) = row_cells.get_mut(col as usize) {
                        cell.hit_count += 1;
                        cell.coverage = 1.0;
                    }
                }
            }
        }
        self.total_interactions += 1;
    }

    /// Record coverage for an element by ID
    pub fn record_element(&mut self, _id: &str, bounds: Region) {
        self.record_region(bounds);
    }

    /// Generate coverage report
    #[must_use]
    pub fn generate_report(&self) -> PixelCoverageReport {
        let total_cells = self.config.grid_cols * self.config.grid_rows;
        let covered_cells = self
            .cells
            .iter()
            .flat_map(|row| row.iter())
            .filter(|cell| cell.is_covered())
            .count() as u32;

        let overall_coverage = if total_cells > 0 {
            covered_cells as f32 / total_cells as f32
        } else {
            0.0
        };

        let min_coverage = self
            .cells
            .iter()
            .flat_map(|row| row.iter())
            .map(|c| c.coverage)
            .fold(f32::MAX, f32::min);

        let max_coverage = self
            .cells
            .iter()
            .flat_map(|row| row.iter())
            .map(|c| c.coverage)
            .fold(0.0_f32, f32::max);

        PixelCoverageReport {
            grid_width: self.config.grid_cols,
            grid_height: self.config.grid_rows,
            overall_coverage,
            covered_cells,
            total_cells,
            min_coverage: if !min_coverage.is_finite() || min_coverage > 1.0 {
                0.0
            } else {
                min_coverage
            },
            max_coverage,
            total_interactions: self.total_interactions,
            meets_threshold: overall_coverage >= self.threshold,
            uncovered_regions: self.find_uncovered_regions(),
        }
    }

    /// Get list of uncovered regions
    #[must_use]
    pub fn uncovered_regions(&self) -> Vec<Region> {
        self.find_uncovered_regions()
    }

    /// Find contiguous uncovered regions
    fn find_uncovered_regions(&self) -> Vec<Region> {
        let mut regions = Vec::new();

        for (row_idx, row) in self.cells.iter().enumerate() {
            for (col_idx, cell) in row.iter().enumerate() {
                if !cell.is_covered() {
                    // Convert to screen coordinates
                    let region = self.config.cell_to_region(col_idx as u32, row_idx as u32);
                    regions.push(region);
                }
            }
        }

        regions
    }

    /// Get cells for rendering
    #[must_use]
    pub fn cells(&self) -> &Vec<Vec<CoverageCell>> {
        &self.cells
    }

    /// Generate terminal heatmap
    #[must_use]
    pub fn terminal_heatmap(&self) -> super::heatmap::TerminalHeatmap {
        super::heatmap::TerminalHeatmap::from_tracker(self)
    }

    /// Generate PNG heatmap exporter
    #[must_use]
    pub fn png_heatmap(&self, width: u32, height: u32) -> super::heatmap::PngHeatmap {
        super::heatmap::PngHeatmap::new(width, height)
    }

    /// Export to PNG bytes with default settings
    #[cfg(feature = "media")]
    pub fn export_png(&self, width: u32, height: u32) -> Result<Vec<u8>, std::io::Error> {
        self.png_heatmap(width, height).export(&self.cells)
    }

    /// Export to PNG file with default settings
    #[cfg(feature = "media")]
    pub fn export_png_to_file(
        &self,
        width: u32,
        height: u32,
        path: &std::path::Path,
    ) -> Result<(), std::io::Error> {
        self.png_heatmap(width, height)
            .export_to_file(&self.cells, path)
    }
}

/// Builder for `PixelCoverageTracker`
#[derive(Debug, Clone)]
pub struct PixelCoverageTrackerBuilder {
    width: u32,
    height: u32,
    grid_cols: u32,
    grid_rows: u32,
    threshold: f32,
}

impl Default for PixelCoverageTrackerBuilder {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            grid_cols: 64,
            grid_rows: 36,
            threshold: 0.8,
        }
    }
}

impl PixelCoverageTrackerBuilder {
    /// Set screen resolution
    #[must_use]
    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set grid size
    #[must_use]
    pub fn grid_size(mut self, cols: u32, rows: u32) -> Self {
        self.grid_cols = cols;
        self.grid_rows = rows;
        self
    }

    /// Set coverage threshold
    #[must_use]
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Build the tracker
    #[must_use]
    pub fn build(self) -> PixelCoverageTracker {
        let mut tracker =
            PixelCoverageTracker::new(self.width, self.height, self.grid_cols, self.grid_rows);
        tracker.threshold = self.threshold;
        tracker
    }
}

/// Pixel coverage report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PixelCoverageReport {
    /// Grid width (columns)
    pub grid_width: u32,
    /// Grid height (rows)
    pub grid_height: u32,
    /// Overall coverage percentage (0.0 - 1.0)
    pub overall_coverage: f32,
    /// Number of covered cells
    pub covered_cells: u32,
    /// Total number of cells
    pub total_cells: u32,
    /// Minimum coverage in any cell
    pub min_coverage: f32,
    /// Maximum coverage in any cell
    pub max_coverage: f32,
    /// Total number of interactions recorded
    pub total_interactions: u64,
    /// Whether coverage meets the threshold
    pub meets_threshold: bool,
    /// List of uncovered regions
    pub uncovered_regions: Vec<Region>,
}

impl Default for PixelCoverageReport {
    fn default() -> Self {
        Self {
            grid_width: 0,
            grid_height: 0,
            overall_coverage: 0.0,
            covered_cells: 0,
            total_cells: 0,
            min_coverage: 0.0,
            max_coverage: 0.0,
            total_interactions: 0,
            meets_threshold: false,
            uncovered_regions: Vec::new(),
        }
    }
}

impl PixelCoverageReport {
    /// Get coverage as percentage (0-100)
    #[must_use]
    pub fn percent(&self) -> f32 {
        self.overall_coverage * 100.0
    }

    /// Check if coverage is complete (100%)
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.covered_cells == self.total_cells
    }
}

/// Line/element coverage report (from GuiCoverage)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LineCoverageReport {
    /// Element coverage percentage (0.0 - 1.0)
    pub element_coverage: f32,
    /// Screen coverage percentage (0.0 - 1.0)
    pub screen_coverage: f32,
    /// Journey coverage percentage (0.0 - 1.0)
    pub journey_coverage: f32,
    /// Total elements tracked
    pub total_elements: usize,
    /// Covered elements
    pub covered_elements: usize,
}

impl LineCoverageReport {
    /// Create from component coverages
    #[must_use]
    pub fn new(
        element_coverage: f32,
        screen_coverage: f32,
        journey_coverage: f32,
        total_elements: usize,
        covered_elements: usize,
    ) -> Self {
        Self {
            element_coverage,
            screen_coverage,
            journey_coverage,
            total_elements,
            covered_elements,
        }
    }

    /// Get average coverage across all dimensions
    #[must_use]
    pub fn average(&self) -> f32 {
        (self.element_coverage + self.screen_coverage + self.journey_coverage) / 3.0
    }
}

/// Combined coverage report (line + pixel coverage)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CombinedCoverageReport {
    /// Line/element coverage (logical)
    pub line_coverage: LineCoverageReport,
    /// Pixel/region coverage (visual)
    pub pixel_coverage: PixelCoverageReport,
    /// Overall score (weighted average)
    pub overall_score: f32,
    /// Meets threshold
    pub meets_threshold: bool,
    /// Weight for line coverage (0.0 - 1.0)
    pub line_weight: f32,
    /// Weight for pixel coverage (0.0 - 1.0)
    pub pixel_weight: f32,
}

impl CombinedCoverageReport {
    /// Default weight for line coverage (50%)
    pub const DEFAULT_LINE_WEIGHT: f32 = 0.5;
    /// Default weight for pixel coverage (50%)
    pub const DEFAULT_PIXEL_WEIGHT: f32 = 0.5;
    /// Default threshold for meeting coverage requirements (80%)
    pub const DEFAULT_THRESHOLD: f32 = 0.8;

    /// Create from line and pixel reports with default weights
    #[must_use]
    pub fn from_parts(line: LineCoverageReport, pixel: PixelCoverageReport) -> Self {
        Self::from_parts_weighted(
            line,
            pixel,
            Self::DEFAULT_LINE_WEIGHT,
            Self::DEFAULT_PIXEL_WEIGHT,
        )
    }

    /// Create from line and pixel reports with custom weights
    #[must_use]
    pub fn from_parts_weighted(
        line: LineCoverageReport,
        pixel: PixelCoverageReport,
        line_weight: f32,
        pixel_weight: f32,
    ) -> Self {
        let line_score = line.element_coverage;
        let pixel_score = pixel.overall_coverage;
        let overall_score = line_score * line_weight + pixel_score * pixel_weight;

        Self {
            line_coverage: line,
            pixel_coverage: pixel,
            overall_score,
            meets_threshold: overall_score >= Self::DEFAULT_THRESHOLD,
            line_weight,
            pixel_weight,
        }
    }

    /// Set threshold and update meets_threshold
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.meets_threshold = self.overall_score >= threshold;
        self
    }

    /// Get line coverage percentage (0-100)
    #[must_use]
    pub fn line_percent(&self) -> f32 {
        self.line_coverage.element_coverage * 100.0
    }

    /// Get pixel coverage percentage (0-100)
    #[must_use]
    pub fn pixel_percent(&self) -> f32 {
        self.pixel_coverage.overall_coverage * 100.0
    }

    /// Get overall score percentage (0-100)
    #[must_use]
    pub fn overall_percent(&self) -> f32 {
        self.overall_score * 100.0
    }

    /// Generate text summary
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Combined Coverage Report\n\
             ========================\n\
             Line Coverage:  {:.1}% ({}/{} elements)\n\
             Pixel Coverage: {:.1}% ({}/{} cells)\n\
             Overall Score:  {:.1}%\n\
             Threshold Met:  {}\n",
            self.line_percent(),
            self.line_coverage.covered_elements,
            self.line_coverage.total_elements,
            self.pixel_percent(),
            self.pixel_coverage.covered_cells,
            self.pixel_coverage.total_cells,
            self.overall_percent(),
            if self.meets_threshold { "✓" } else { "✗" }
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_config_cell_dimensions() {
        let config = GridConfig {
            screen_width: 1920,
            screen_height: 1080,
            grid_cols: 64,
            grid_rows: 36,
        };

        assert_eq!(config.cell_width(), 30);
        assert_eq!(config.cell_height(), 30);
    }

    #[test]
    fn test_grid_config_point_to_cell() {
        let config = GridConfig {
            screen_width: 100,
            screen_height: 100,
            grid_cols: 10,
            grid_rows: 10,
        };

        assert_eq!(config.point_to_cell(Point::new(0, 0)), (0, 0));
        assert_eq!(config.point_to_cell(Point::new(15, 25)), (1, 2));
        assert_eq!(config.point_to_cell(Point::new(99, 99)), (9, 9));
    }

    #[test]
    fn test_coverage_cell_default() {
        let cell = CoverageCell::default();
        assert_eq!(cell.hit_count, 0);
        assert!(!cell.is_covered());
    }

    #[test]
    fn test_builder_defaults() {
        let builder = PixelCoverageTrackerBuilder::default();
        assert_eq!(builder.width, 1920);
        assert_eq!(builder.height, 1080);
        assert_eq!(builder.grid_cols, 64);
        assert_eq!(builder.grid_rows, 36);
    }

    #[test]
    fn test_report_percent() {
        let report = PixelCoverageReport {
            grid_width: 10,
            grid_height: 10,
            overall_coverage: 0.75,
            covered_cells: 75,
            total_cells: 100,
            min_coverage: 0.0,
            max_coverage: 1.0,
            total_interactions: 100,
            meets_threshold: false,
            uncovered_regions: vec![],
        };

        assert!((report.percent() - 75.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_report_is_complete() {
        let complete = PixelCoverageReport {
            grid_width: 10,
            grid_height: 10,
            overall_coverage: 1.0,
            covered_cells: 100,
            total_cells: 100,
            min_coverage: 1.0,
            max_coverage: 1.0,
            total_interactions: 100,
            meets_threshold: true,
            uncovered_regions: vec![],
        };

        assert!(complete.is_complete());

        let incomplete = PixelCoverageReport {
            covered_cells: 99,
            total_cells: 100,
            ..complete
        };

        assert!(!incomplete.is_complete());
    }

    #[test]
    fn test_region_area() {
        let region = Region::new(0, 0, 100, 50);
        assert_eq!(region.area(), 5000);
    }

    // =========================================================================
    // Combined Coverage Report Tests
    // =========================================================================

    #[test]
    fn h0_combined_01_from_parts() {
        let line_report = LineCoverageReport::new(0.90, 1.0, 0.80, 22, 20);
        let pixel_report = PixelCoverageReport {
            overall_coverage: 0.85,
            ..Default::default()
        };

        let combined = CombinedCoverageReport::from_parts(line_report, pixel_report);

        // Weighted average: (0.90 * 0.5 + 0.85 * 0.5) = 0.875
        assert!((combined.overall_score - 0.875).abs() < 0.01);
        assert!(combined.meets_threshold);
    }

    #[test]
    fn h0_combined_02_custom_weights() {
        let line_report = LineCoverageReport::new(1.0, 1.0, 1.0, 10, 10);
        let pixel_report = PixelCoverageReport {
            overall_coverage: 0.0,
            ..Default::default()
        };

        // 100% line weight, 0% pixel weight
        let combined =
            CombinedCoverageReport::from_parts_weighted(line_report, pixel_report, 1.0, 0.0);

        assert!((combined.overall_score - 1.0).abs() < 0.01);
    }

    #[test]
    fn h0_combined_03_threshold() {
        let line_report = LineCoverageReport::new(0.5, 0.5, 0.5, 10, 5);
        let pixel_report = PixelCoverageReport {
            overall_coverage: 0.5,
            ..Default::default()
        };

        let combined = CombinedCoverageReport::from_parts(line_report, pixel_report);

        assert!(!combined.meets_threshold); // 0.5 < 0.8
        assert!((combined.overall_score - 0.5).abs() < 0.01);

        let relaxed = combined.with_threshold(0.4);
        assert!(relaxed.meets_threshold);
    }

    #[test]
    fn h0_combined_04_summary() {
        let line_report = LineCoverageReport::new(0.90, 1.0, 0.80, 22, 20);
        let pixel_report = PixelCoverageReport {
            overall_coverage: 0.85,
            covered_cells: 85,
            total_cells: 100,
            ..Default::default()
        };

        let combined = CombinedCoverageReport::from_parts(line_report, pixel_report);
        let summary = combined.summary();

        assert!(summary.contains("Line Coverage"));
        assert!(summary.contains("Pixel Coverage"));
        assert!(summary.contains("Overall Score"));
        assert!(summary.contains("✓"));
    }

    #[test]
    fn h0_combined_05_line_report_average() {
        let report = LineCoverageReport::new(0.9, 0.6, 0.9, 10, 8);
        assert!((report.average() - 0.8).abs() < 0.01);
    }

    #[test]
    fn h0_combined_06_percentages() {
        let line_report = LineCoverageReport::new(0.90, 1.0, 0.80, 22, 20);
        let pixel_report = PixelCoverageReport {
            overall_coverage: 0.85,
            ..Default::default()
        };

        let combined = CombinedCoverageReport::from_parts(line_report, pixel_report);

        assert!((combined.line_percent() - 90.0).abs() < 0.01);
        assert!((combined.pixel_percent() - 85.0).abs() < 0.01);
        assert!((combined.overall_percent() - 87.5).abs() < 0.01);
    }

    #[test]
    fn h0_combined_07_default() {
        let combined = CombinedCoverageReport::default();
        assert_eq!(combined.overall_score, 0.0);
        assert!(!combined.meets_threshold);
    }

    // =========================================================================
    // PNG export convenience tests
    // =========================================================================

    #[test]
    fn h0_tracker_png_export() {
        let mut tracker = PixelCoverageTracker::new(100, 100, 10, 10);
        tracker.record_region(Region::new(0, 0, 50, 50));

        let png = tracker.export_png(200, 200).unwrap();
        assert!(!png.is_empty());
        // Verify PNG header
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn h0_tracker_png_heatmap() {
        let tracker = PixelCoverageTracker::new(100, 100, 10, 10);
        let heatmap = tracker.png_heatmap(200, 200);
        let png = heatmap.export(tracker.cells()).unwrap();
        assert!(!png.is_empty());
    }
}
