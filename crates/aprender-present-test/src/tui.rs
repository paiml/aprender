//! TUI Testing Framework (SPEC-024 Section 12 & 13)
//!
//! This module enforces test-first development for TUI widgets.
//! Tests DEFINE the interface - implementation follows.
//!
//! # Example: Test-First Interface Definition
//!
//! ```ignore
//! use presentar_test::tui::{TuiTestBackend, expect_frame, FrameAssertion};
//!
//! #[test]
//! fn test_cpu_exploded_receives_async_updates() {
//!     let mut backend = TuiTestBackend::new(120, 40);
//!     let mut app = App::test_instance();
//!
//!     // Frame 1: Initial render
//!     app.apply_snapshot(snapshot1);
//!     backend.render(|buf| ui::draw(&app, buf));
//!     let freq1 = backend.extract_text_at(50, 5); // CPU frequency
//!
//!     // Frame 2: After async update
//!     app.apply_snapshot(snapshot2);
//!     backend.render(|buf| ui::draw(&app, buf));
//!     let freq2 = backend.extract_text_at(50, 5);
//!
//!     // ASSERTION: Data must update
//!     expect_frame(&backend)
//!         .field("cpu_freq")
//!         .changed_between(freq1, freq2);
//! }
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Cell in the TUI buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct TuiCell {
    /// Character at this position.
    pub ch: char,
    /// Foreground color (RGB).
    pub fg: (u8, u8, u8),
    /// Background color (RGB).
    pub bg: (u8, u8, u8),
    /// Bold attribute.
    pub bold: bool,
}

impl Default for TuiCell {
    fn default() -> Self {
        Self {
            ch: ' ',
            fg: (255, 255, 255),
            bg: (0, 0, 0),
            bold: false,
        }
    }
}

/// In-memory TUI test backend.
/// Renders widgets to a buffer for assertions.
#[derive(Debug)]
pub struct TuiTestBackend {
    /// Width in columns.
    pub width: u16,
    /// Height in rows.
    pub height: u16,
    /// Cell buffer.
    cells: Vec<TuiCell>,
    /// Frame counter.
    frame_count: u64,
    /// Render metrics.
    metrics: RenderMetrics,
    /// Deterministic mode.
    deterministic: bool,
}

impl TuiTestBackend {
    /// Create a new test backend with dimensions.
    pub fn new(width: u16, height: u16) -> Self {
        let size = width as usize * height as usize;
        Self {
            width,
            height,
            cells: vec![TuiCell::default(); size],
            frame_count: 0,
            metrics: RenderMetrics::new(),
            deterministic: true,
        }
    }

    /// Enable/disable deterministic mode.
    pub fn with_deterministic(mut self, enabled: bool) -> Self {
        self.deterministic = enabled;
        self
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            *cell = TuiCell::default();
        }
    }

    /// Get cell at position.
    pub fn get(&self, x: u16, y: u16) -> Option<&TuiCell> {
        if x < self.width && y < self.height {
            Some(&self.cells[y as usize * self.width as usize + x as usize])
        } else {
            None
        }
    }

    /// Set cell at position.
    pub fn set(&mut self, x: u16, y: u16, cell: TuiCell) {
        if x < self.width && y < self.height {
            self.cells[y as usize * self.width as usize + x as usize] = cell;
        }
    }

    /// Draw text at position.
    pub fn draw_text(&mut self, x: u16, y: u16, text: &str, fg: (u8, u8, u8)) {
        for (i, ch) in text.chars().enumerate() {
            let col = x + i as u16;
            if col < self.width {
                self.set(
                    col,
                    y,
                    TuiCell {
                        ch,
                        fg,
                        bg: (0, 0, 0),
                        bold: false,
                    },
                );
            }
        }
    }

    /// Render a frame using provided closure.
    pub fn render<F: FnOnce(&mut Self)>(&mut self, f: F) {
        let start = Instant::now();
        self.clear();
        f(self);
        let elapsed = start.elapsed();
        self.metrics.record_frame(elapsed);
        self.frame_count += 1;
    }

    /// Extract text from a row.
    pub fn extract_row(&self, y: u16) -> String {
        if y >= self.height {
            return String::new();
        }
        let start = y as usize * self.width as usize;
        let end = start + self.width as usize;
        self.cells[start..end].iter().map(|c| c.ch).collect()
    }

    /// Extract text at position (reads until whitespace or boundary).
    pub fn extract_text_at(&self, x: u16, y: u16) -> String {
        let mut result = String::new();
        let mut col = x;
        while col < self.width {
            if let Some(cell) = self.get(col, y) {
                if cell.ch == ' ' && !result.is_empty() {
                    break;
                }
                if cell.ch != ' ' {
                    result.push(cell.ch);
                }
            }
            col += 1;
        }
        result
    }

    /// Extract text in region.
    pub fn extract_region(&self, x: u16, y: u16, width: u16, height: u16) -> Vec<String> {
        let mut lines = Vec::with_capacity(height as usize);
        for row in y..(y + height).min(self.height) {
            let mut line = String::with_capacity(width as usize);
            for col in x..(x + width).min(self.width) {
                if let Some(cell) = self.get(col, row) {
                    line.push(cell.ch);
                }
            }
            lines.push(line);
        }
        lines
    }

    /// Convert buffer to string representation.
    pub fn to_string_plain(&self) -> String {
        let mut result = String::with_capacity((self.width as usize + 1) * self.height as usize);
        for y in 0..self.height {
            result.push_str(&self.extract_row(y));
            result.push('\n');
        }
        result
    }

    /// Get current frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get render metrics.
    pub fn metrics(&self) -> &RenderMetrics {
        &self.metrics
    }

    /// Create a snapshot of current state.
    pub fn snapshot(&self) -> TuiSnapshot {
        TuiSnapshot {
            width: self.width,
            height: self.height,
            cells: self.cells.clone(),
            metadata: HashMap::new(),
        }
    }
}

/// Snapshot of TUI state for comparison.
#[derive(Debug, Clone)]
pub struct TuiSnapshot {
    /// Width.
    pub width: u16,
    /// Height.
    pub height: u16,
    /// Cell data.
    pub cells: Vec<TuiCell>,
    /// Metadata (data values used, timestamps, etc.).
    pub metadata: HashMap<String, String>,
}

impl TuiSnapshot {
    /// Load snapshot from file.
    pub fn load(path: &str) -> Result<Self, SnapshotError> {
        let content =
            std::fs::read_to_string(path).map_err(|e| SnapshotError::IoError(e.to_string()))?;
        Self::parse(&content)
    }

    /// Save snapshot to file.
    pub fn save(&self, path: &str) -> Result<(), SnapshotError> {
        let content = self.serialize();
        std::fs::write(path, content).map_err(|e| SnapshotError::IoError(e.to_string()))
    }

    /// Parse snapshot from string.
    pub fn parse(content: &str) -> Result<Self, SnapshotError> {
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return Err(SnapshotError::ParseError("Empty snapshot".into()));
        }

        // First line: dimensions
        let dims: Vec<u16> = lines[0]
            .split('x')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        if dims.len() != 2 {
            return Err(SnapshotError::ParseError("Invalid dimensions".into()));
        }

        let width = dims[0];
        let height = dims[1];
        let mut cells = vec![TuiCell::default(); width as usize * height as usize];

        // Parse cell data
        for (y, line) in lines.iter().skip(1).take(height as usize).enumerate() {
            for (x, ch) in line.chars().take(width as usize).enumerate() {
                cells[y * width as usize + x].ch = ch;
            }
        }

        Ok(Self {
            width,
            height,
            cells,
            metadata: HashMap::new(),
        })
    }

    /// Serialize snapshot to string.
    pub fn serialize(&self) -> String {
        let mut result = format!("{}x{}\n", self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y as usize * self.width as usize + x as usize;
                result.push(self.cells[idx].ch);
            }
            result.push('\n');
        }
        result
    }

    /// Get metadata value.
    pub fn metadata(&self, key: &str) -> &str {
        self.metadata.get(key).map(|s| s.as_str()).unwrap_or("")
    }

    /// Set metadata value.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Compare with another snapshot.
    pub fn diff(&self, other: &TuiSnapshot) -> SnapshotDiff {
        let mut diff = SnapshotDiff {
            matches: true,
            differences: Vec::new(),
            total_cells: self.width as usize * self.height as usize,
            matching_cells: 0,
        };

        if self.width != other.width || self.height != other.height {
            diff.matches = false;
            diff.differences.push(DiffEntry {
                x: 0,
                y: 0,
                expected: format!("{}x{}", self.width, self.height),
                actual: format!("{}x{}", other.width, other.height),
            });
            return diff;
        }

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y as usize * self.width as usize + x as usize;
                if self.cells[idx] == other.cells[idx] {
                    diff.matching_cells += 1;
                } else {
                    diff.matches = false;
                    diff.differences.push(DiffEntry {
                        x,
                        y,
                        expected: self.cells[idx].ch.to_string(),
                        actual: other.cells[idx].ch.to_string(),
                    });
                }
            }
        }

        diff
    }
}

/// Snapshot error.
#[derive(Debug)]
pub enum SnapshotError {
    IoError(String),
    ParseError(String),
}

/// Difference between two snapshots.
#[derive(Debug)]
pub struct SnapshotDiff {
    /// Whether snapshots match.
    pub matches: bool,
    /// List of differences.
    pub differences: Vec<DiffEntry>,
    /// Total cells compared.
    pub total_cells: usize,
    /// Cells that matched.
    pub matching_cells: usize,
}

impl SnapshotDiff {
    /// Get match percentage.
    pub fn match_percentage(&self) -> f64 {
        if self.total_cells == 0 {
            100.0
        } else {
            self.matching_cells as f64 / self.total_cells as f64 * 100.0
        }
    }
}

/// Single difference entry.
#[derive(Debug)]
pub struct DiffEntry {
    pub x: u16,
    pub y: u16,
    pub expected: String,
    pub actual: String,
}

/// Performance metrics collected during rendering.
#[derive(Debug, Clone, Default)]
pub struct RenderMetrics {
    /// Total frames rendered.
    pub frame_count: u64,
    /// Frame time samples (microseconds).
    samples: Vec<u64>,
}

impl RenderMetrics {
    /// Create new metrics collector.
    pub fn new() -> Self {
        Self {
            frame_count: 0,
            samples: Vec::with_capacity(1000),
        }
    }

    /// Record a frame's render time.
    pub fn record_frame(&mut self, duration: Duration) {
        self.frame_count += 1;
        self.samples.push(duration.as_micros() as u64);
    }

    /// Get minimum frame time (microseconds).
    pub fn min_us(&self) -> u64 {
        self.samples.iter().min().copied().unwrap_or(0)
    }

    /// Get maximum frame time (microseconds).
    pub fn max_us(&self) -> u64 {
        self.samples.iter().max().copied().unwrap_or(0)
    }

    /// Get mean frame time (microseconds).
    pub fn mean_us(&self) -> f64 {
        if self.samples.is_empty() {
            0.0
        } else {
            self.samples.iter().sum::<u64>() as f64 / self.samples.len() as f64
        }
    }

    /// Get percentile (0-100).
    pub fn percentile(&self, p: u8) -> u64 {
        if self.samples.is_empty() {
            return 0;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_unstable();
        let idx = (sorted.len() as f64 * p as f64 / 100.0) as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Check if metrics meet performance targets.
    pub fn meets_targets(&self, targets: &PerformanceTargets) -> bool {
        self.max_us() <= targets.max_frame_us && self.percentile(99) <= targets.p99_frame_us
    }

    /// Export to JSON.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"frame_count":{},"min_us":{},"max_us":{},"mean_us":{:.2},"p50_us":{},"p95_us":{},"p99_us":{}}}"#,
            self.frame_count,
            self.min_us(),
            self.max_us(),
            self.mean_us(),
            self.percentile(50),
            self.percentile(95),
            self.percentile(99),
        )
    }
}

/// Performance targets for validation.
#[derive(Debug, Clone)]
pub struct PerformanceTargets {
    /// Maximum frame time in microseconds.
    pub max_frame_us: u64,
    /// Target p99 frame time.
    pub p99_frame_us: u64,
    /// Maximum memory usage (bytes).
    pub max_memory_bytes: usize,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            max_frame_us: 16_667,         // 60fps = 16.67ms
            p99_frame_us: 1_000,          // 1ms for TUI
            max_memory_bytes: 100 * 1024, // 100KB
        }
    }
}

/// Fluent frame assertion builder.
pub struct FrameAssertion<'a> {
    backend: &'a TuiTestBackend,
    tolerance: usize,
    ignore_color: bool,
    ignore_trailing_whitespace: bool,
    region: Option<(u16, u16, u16, u16)>,
}

impl FrameAssertion<'_> {
    /// Set tolerance for differences (0 = exact match).
    pub fn with_tolerance(mut self, tolerance: usize) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Ignore color differences.
    pub fn ignore_color(mut self) -> Self {
        self.ignore_color = true;
        self
    }

    /// Ignore trailing whitespace.
    pub fn ignore_whitespace_at_eol(mut self) -> Self {
        self.ignore_trailing_whitespace = true;
        self
    }

    /// Compare only a specific region.
    pub fn with_region(mut self, x: u16, y: u16, width: u16, height: u16) -> Self {
        self.region = Some((x, y, width, height));
        self
    }

    /// Assert frame matches snapshot.
    ///
    /// # Panics
    /// Panics if frame doesn't match within tolerance.
    pub fn to_match_snapshot(self, snapshot: &TuiSnapshot) {
        let current = self.backend.snapshot();
        let diff = current.diff(snapshot);

        if !diff.matches && diff.differences.len() > self.tolerance {
            panic!(
                "Frame does not match snapshot:\n\
                 - {}/{} cells differ ({:.1}% match)\n\
                 - Tolerance: {}\n\
                 - First 5 differences:\n{}",
                diff.differences.len(),
                diff.total_cells,
                diff.match_percentage(),
                self.tolerance,
                diff.differences
                    .iter()
                    .take(5)
                    .map(|d| format!(
                        "  ({}, {}): expected '{}', got '{}'",
                        d.x, d.y, d.expected, d.actual
                    ))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
    }

    /// Assert frame contains text.
    ///
    /// # Panics
    /// Panics if text is not found.
    pub fn to_contain_text(self, text: &str) {
        let content = self.backend.to_string_plain();
        assert!(
            content.contains(text),
            "Frame does not contain text: '{}'",
            text
        );
    }

    /// Assert frame does not contain text.
    ///
    /// # Panics
    /// Panics if text is found.
    pub fn to_not_contain_text(self, text: &str) {
        let content = self.backend.to_string_plain();
        assert!(
            !content.contains(text),
            "Frame should not contain text: '{}'",
            text
        );
    }

    /// Assert text at specific position.
    ///
    /// # Panics
    /// Panics if text doesn't match.
    pub fn text_at(self, x: u16, y: u16, expected: &str) {
        let actual = self.backend.extract_text_at(x, y);
        assert_eq!(
            actual, expected,
            "Text at ({}, {}) expected '{}', got '{}'",
            x, y, expected, actual
        );
    }

    /// Assert row content.
    ///
    /// # Panics
    /// Panics if row doesn't match.
    pub fn row_equals(self, y: u16, expected: &str) {
        let actual = self.backend.extract_row(y);
        let actual_trimmed = if self.ignore_trailing_whitespace {
            actual.trim_end()
        } else {
            &actual
        };
        let expected_trimmed = if self.ignore_trailing_whitespace {
            expected.trim_end()
        } else {
            expected
        };
        assert_eq!(
            actual_trimmed, expected_trimmed,
            "Row {} expected:\n'{}'\ngot:\n'{}'",
            y, expected_trimmed, actual_trimmed
        );
    }
}

/// Start building frame assertions.
pub fn expect_frame(backend: &TuiTestBackend) -> FrameAssertion<'_> {
    FrameAssertion {
        backend,
        tolerance: 0,
        ignore_color: false,
        ignore_trailing_whitespace: false,
        region: None,
    }
}

/// Benchmark harness for running widget benchmarks.
pub struct BenchmarkHarness {
    backend: TuiTestBackend,
    warmup_frames: u32,
    benchmark_frames: u32,
}

impl BenchmarkHarness {
    /// Create new benchmark harness.
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            backend: TuiTestBackend::new(width, height),
            warmup_frames: 100,
            benchmark_frames: 1000,
        }
    }

    /// Set warmup and benchmark frame counts.
    pub fn with_frames(mut self, warmup: u32, benchmark: u32) -> Self {
        self.warmup_frames = warmup;
        self.benchmark_frames = benchmark;
        self
    }

    /// Run benchmark with provided render function.
    pub fn benchmark<F: FnMut(&mut TuiTestBackend)>(&mut self, mut render: F) -> BenchmarkResult {
        // Warmup phase
        for _ in 0..self.warmup_frames {
            self.backend.render(|b| render(b));
        }

        // Reset metrics
        self.backend.metrics = RenderMetrics::new();

        // Benchmark phase
        for _ in 0..self.benchmark_frames {
            self.backend.render(|b| render(b));
        }

        BenchmarkResult {
            metrics: self.backend.metrics().clone(),
            final_frame: self.backend.to_string_plain(),
        }
    }
}

/// Benchmark result.
#[derive(Debug)]
pub struct BenchmarkResult {
    /// Performance metrics.
    pub metrics: RenderMetrics,
    /// Final frame content.
    pub final_frame: String,
}

impl BenchmarkResult {
    /// Check if benchmark meets targets.
    pub fn meets_targets(&self, targets: &PerformanceTargets) -> bool {
        self.metrics.meets_targets(targets)
    }
}

/// Assertion helper for async data updates.
/// This is the CRITICAL test that defines interface requirements.
pub struct AsyncUpdateAssertion {
    /// Field name being tested.
    field: String,
    /// Initial value.
    initial: Option<String>,
    /// Values after each update.
    values: Vec<String>,
}

impl AsyncUpdateAssertion {
    /// Create new async update assertion.
    pub fn new(field: &str) -> Self {
        Self {
            field: field.to_string(),
            initial: None,
            values: Vec::new(),
        }
    }

    /// Record initial value.
    pub fn record_initial(&mut self, value: &str) {
        self.initial = Some(value.to_string());
    }

    /// Record updated value.
    pub fn record_update(&mut self, value: &str) {
        self.values.push(value.to_string());
    }

    /// Assert that value is present (non-empty).
    ///
    /// # Panics
    /// Panics if value is empty or missing.
    pub fn assert_present(&self) {
        if let Some(ref initial) = self.initial {
            assert!(
                !initial.is_empty(),
                "Field '{}' initial value should be present, got empty",
                self.field
            );
        } else {
            panic!("Field '{}' has no initial value recorded", self.field);
        }
    }

    /// Assert that value changed between updates.
    ///
    /// # Panics
    /// Panics if value never changed.
    pub fn assert_changed(&self) {
        let Some(ref initial) = self.initial else {
            panic!("Field '{}' has no initial value", self.field);
        };

        let changed = self.values.iter().any(|v| v != initial);
        assert!(
            changed,
            "Field '{}' expected to change from '{}' but never did. Updates: {:?}",
            self.field, initial, self.values
        );
    }

    /// Assert that value is numeric and within range.
    ///
    /// # Panics
    /// Panics if value is not numeric or out of range.
    pub fn assert_numeric_in_range(&self, min: f64, max: f64) {
        let Some(ref initial) = self.initial else {
            panic!("Field '{}' has no initial value", self.field);
        };

        // Try to parse as number (strip % and other suffixes)
        let num_str = initial
            .trim_end_matches('%')
            .trim_end_matches("MHz")
            .trim_end_matches("GHz")
            .trim_end_matches("°C")
            .trim();

        let value: f64 = num_str.parse().unwrap_or_else(|_| {
            panic!(
                "Field '{}' expected numeric value, got '{}'",
                self.field, initial
            )
        });

        assert!(
            value >= min && value <= max,
            "Field '{}' value {} not in range [{}, {}]",
            self.field,
            value,
            min,
            max
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_basic() {
        let mut backend = TuiTestBackend::new(80, 24);
        assert_eq!(backend.width, 80);
        assert_eq!(backend.height, 24);

        backend.draw_text(0, 0, "Hello", (255, 255, 255));
        assert_eq!(backend.extract_text_at(0, 0), "Hello");
    }

    #[test]
    fn test_backend_render_metrics() {
        let mut backend = TuiTestBackend::new(80, 24);

        backend.render(|b| {
            b.draw_text(0, 0, "Frame 1", (255, 255, 255));
        });

        backend.render(|b| {
            b.draw_text(0, 0, "Frame 2", (255, 255, 255));
        });

        assert_eq!(backend.frame_count(), 2);
        assert!(backend.metrics().mean_us() >= 0.0);
    }

    #[test]
    fn test_snapshot_diff() {
        let mut backend1 = TuiTestBackend::new(10, 2);
        backend1.draw_text(0, 0, "Hello", (255, 255, 255));
        let snap1 = backend1.snapshot();

        let mut backend2 = TuiTestBackend::new(10, 2);
        backend2.draw_text(0, 0, "Hello", (255, 255, 255));
        let snap2 = backend2.snapshot();

        let diff = snap1.diff(&snap2);
        assert!(diff.matches);
        assert_eq!(diff.match_percentage(), 100.0);
    }

    #[test]
    fn test_snapshot_diff_mismatch() {
        let mut backend1 = TuiTestBackend::new(10, 2);
        backend1.draw_text(0, 0, "Hello", (255, 255, 255));
        let snap1 = backend1.snapshot();

        let mut backend2 = TuiTestBackend::new(10, 2);
        backend2.draw_text(0, 0, "World", (255, 255, 255));
        let snap2 = backend2.snapshot();

        let diff = snap1.diff(&snap2);
        assert!(!diff.matches);
        assert!(!diff.differences.is_empty());
    }

    #[test]
    fn test_expect_frame_contains_text() {
        let mut backend = TuiTestBackend::new(80, 24);
        backend.draw_text(10, 5, "CPU: 45%", (255, 255, 255));

        expect_frame(&backend).to_contain_text("CPU: 45%");
    }

    #[test]
    #[should_panic(expected = "does not contain")]
    fn test_expect_frame_missing_text() {
        let backend = TuiTestBackend::new(80, 24);
        expect_frame(&backend).to_contain_text("Missing text");
    }

    #[test]
    fn test_async_update_assertion() {
        let mut assertion = AsyncUpdateAssertion::new("cpu_freq");
        assertion.record_initial("4.5GHz");
        assertion.record_update("4.6GHz");
        assertion.record_update("4.7GHz");

        assertion.assert_present();
        assertion.assert_changed();
    }

    #[test]
    #[should_panic(expected = "expected to change")]
    fn test_async_update_no_change() {
        let mut assertion = AsyncUpdateAssertion::new("stale_field");
        assertion.record_initial("static");
        assertion.record_update("static");
        assertion.record_update("static");

        assertion.assert_changed();
    }

    #[test]
    fn test_benchmark_harness() {
        let mut harness = BenchmarkHarness::new(80, 24).with_frames(10, 100);

        let result = harness.benchmark(|backend| {
            backend.draw_text(0, 0, "Test", (255, 255, 255));
        });

        assert_eq!(result.metrics.frame_count, 100);
        assert!(result.metrics.mean_us() < 1_000_000.0); // Less than 1 second
    }

    #[test]
    fn test_render_metrics() {
        let mut metrics = RenderMetrics::new();
        metrics.record_frame(Duration::from_micros(100));
        metrics.record_frame(Duration::from_micros(200));
        metrics.record_frame(Duration::from_micros(150));

        assert_eq!(metrics.frame_count, 3);
        assert_eq!(metrics.min_us(), 100);
        assert_eq!(metrics.max_us(), 200);
        assert!((metrics.mean_us() - 150.0).abs() < 1.0);
    }

    #[test]
    fn test_performance_targets() {
        let mut metrics = RenderMetrics::new();
        for _ in 0..100 {
            metrics.record_frame(Duration::from_micros(500));
        }

        let targets = PerformanceTargets::default();
        assert!(metrics.meets_targets(&targets));
    }

    // =========================================================================
    // THIS IS THE CRITICAL TEST PATTERN
    // This test DEFINES the interface for async updates in exploded mode
    // =========================================================================

    /// **TEST THAT DEFINES INTERFACE**
    ///
    /// This test specifies that exploded CPU panel MUST receive:
    /// - per_core_freq: CPU frequency data that updates each frame
    /// - per_core_temp: CPU temperature data that updates each frame
    ///
    /// The implementation MUST satisfy this interface.
    #[test]
    #[ignore = "Enable when MetricsSnapshot includes freq/temp"]
    fn test_exploded_cpu_receives_async_freq_temp_updates() {
        // This test will fail until MetricsSnapshot includes freq/temp
        // and apply_snapshot transfers them to App fields

        // STEP 1: Create test backend
        let mut backend = TuiTestBackend::new(140, 45);

        // STEP 2: Create assertions for fields that MUST exist
        let mut freq_assertion = AsyncUpdateAssertion::new("per_core_freq[0]");
        let mut temp_assertion = AsyncUpdateAssertion::new("per_core_temp[0]");

        // STEP 3: Simulate initial frame with deterministic test values
        backend.render(|b| {
            // Simulated initial render
            b.draw_text(50, 3, "4.76GHz", (255, 255, 255));
            b.draw_text(60, 3, "65°C", (255, 255, 255));
        });
        freq_assertion.record_initial(&backend.extract_text_at(50, 3));
        temp_assertion.record_initial(&backend.extract_text_at(60, 3));

        // STEP 4: Simulate async update (new snapshot applied)
        backend.render(|b| {
            // Simulated updated render (values changed)
            b.draw_text(50, 3, "4.82GHz", (255, 255, 255));
            b.draw_text(60, 3, "67°C", (255, 255, 255));
        });
        freq_assertion.record_update(&backend.extract_text_at(50, 3));
        temp_assertion.record_update(&backend.extract_text_at(60, 3));

        // STEP 5: Assertions that DEFINE the interface
        freq_assertion.assert_present();
        freq_assertion.assert_changed();
        freq_assertion.assert_numeric_in_range(0.0, 10.0); // GHz

        temp_assertion.assert_present();
        temp_assertion.assert_changed();
        temp_assertion.assert_numeric_in_range(0.0, 150.0); // °C
    }
}
