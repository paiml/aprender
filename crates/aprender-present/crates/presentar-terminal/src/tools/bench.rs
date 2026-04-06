//! Headless Benchmarking Tool for cbtop widgets.
//!
//! This module provides automated performance testing, CI/CD integration,
//! and deterministic output capture without requiring a terminal display.
//!
//! # Features
//!
//! - **`HeadlessCanvas`**: In-memory rendering without terminal I/O
//! - **`RenderMetrics`**: Frame time statistics with p50/p95/p99 percentiles
//! - **`BenchmarkHarness`**: Warmup and benchmark phases with comparison support
//! - **`PerformanceTargets`**: Validation against configurable thresholds
//! - **`DeterministicContext`**: Reproducible benchmarks with fixed RNG/timestamps
//!
//! # Example
//!
//! ```ignore
//! use presentar_terminal::tools::bench::{BenchmarkHarness, PerformanceTargets};
//! use presentar_terminal::CpuGrid;
//!
//! let mut harness = BenchmarkHarness::new(80, 24).with_frames(100, 1000);
//! let mut grid = CpuGrid::new(vec![50.0; 48]).with_columns(8).compact();
//! let result = harness.benchmark(&mut grid, Rect::new(0.0, 0.0, 80.0, 10.0));
//!
//! assert!(result.metrics.meets_targets(&PerformanceTargets::default()));
//! ```

use crate::direct::{CellBuffer, Modifiers};
use presentar_core::{Canvas, Color, FontWeight, Point, Rect, TextStyle, Transform2D, Widget};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ============================================================================
// HeadlessCanvas
// ============================================================================

/// In-memory canvas for headless rendering.
///
/// No terminal I/O - pure computation for benchmarking.
/// Implements the `Canvas` trait so widgets can paint to it directly.
#[derive(Debug)]
pub struct HeadlessCanvas {
    /// Cell buffer (same as `DirectTerminalCanvas`).
    buffer: CellBuffer,
    /// Frame counter.
    frame_count: u64,
    /// Metrics collector.
    metrics: RenderMetrics,
    /// Deterministic mode (fixed RNG seeds, timestamps).
    deterministic: bool,
    /// Current foreground color.
    current_fg: Color,
    /// Current background color (reserved for future use).
    #[allow(dead_code)]
    current_bg: Color,
}

impl HeadlessCanvas {
    /// Create headless canvas with dimensions.
    #[must_use]
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            buffer: CellBuffer::new(width, height),
            frame_count: 0,
            metrics: RenderMetrics::new(),
            deterministic: false,
            current_fg: Color::WHITE,
            current_bg: Color::TRANSPARENT,
        }
    }

    /// Enable deterministic mode for reproducible output.
    #[must_use]
    pub fn with_deterministic(mut self, enabled: bool) -> Self {
        self.deterministic = enabled;
        self
    }

    /// Check if in deterministic mode.
    #[must_use]
    pub const fn is_deterministic(&self) -> bool {
        self.deterministic
    }

    /// Render a frame and collect metrics.
    pub fn render_frame<F: FnOnce(&mut Self)>(&mut self, render: F) {
        contract_pre_render!();
        let start = Instant::now();

        self.buffer.clear();
        render(self);

        let elapsed = start.elapsed();
        self.metrics.record_frame(elapsed);
        self.frame_count += 1;
    }

    /// Get the underlying buffer.
    #[must_use]
    pub fn buffer(&self) -> &CellBuffer {
        &self.buffer
    }

    /// Get mutable buffer reference.
    pub fn buffer_mut(&mut self) -> &mut CellBuffer {
        &mut self.buffer
    }

    /// Dump buffer to string (for snapshots).
    #[must_use]
    pub fn dump(&self) -> String {
        let mut output = String::new();
        for y in 0..self.buffer.height() {
            for x in 0..self.buffer.width() {
                if let Some(cell) = self.buffer.get(x, y) {
                    output.push_str(&cell.symbol);
                }
            }
            output.push('\n');
        }
        output
    }

    /// Get collected metrics.
    #[must_use]
    pub fn metrics(&self) -> &RenderMetrics {
        &self.metrics
    }

    /// Get mutable metrics reference.
    pub fn metrics_mut(&mut self) -> &mut RenderMetrics {
        &mut self.metrics
    }

    /// Reset metrics for new benchmark run.
    pub fn reset_metrics(&mut self) {
        self.metrics = RenderMetrics::new();
        self.frame_count = 0;
    }

    /// Get frame count.
    #[must_use]
    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get buffer width.
    #[must_use]
    pub fn width(&self) -> u16 {
        self.buffer.width()
    }

    /// Get buffer height.
    #[must_use]
    pub fn height(&self) -> u16 {
        self.buffer.height()
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

impl Canvas for HeadlessCanvas {
    fn fill_rect(&mut self, rect: Rect, color: Color) {
        let x = rect.x.max(0.0) as u16;
        let y = rect.y.max(0.0) as u16;
        let w = rect.width.max(0.0) as u16;
        let h = rect.height.max(0.0) as u16;
        self.buffer.fill_rect(x, y, w, h, self.current_fg, color);
    }

    fn stroke_rect(&mut self, rect: Rect, color: Color, _width: f32) {
        let x = rect.x.max(0.0) as u16;
        let y = rect.y.max(0.0) as u16;
        let w = rect.width.max(0.0) as u16;
        let h = rect.height.max(0.0) as u16;

        // Top and bottom borders
        for cx in x..x.saturating_add(w).min(self.buffer.width()) {
            self.buffer
                .update(cx, y, "─", color, Color::TRANSPARENT, Modifiers::NONE);
            if h > 0 {
                self.buffer.update(
                    cx,
                    y.saturating_add(h - 1).min(self.buffer.height() - 1),
                    "─",
                    color,
                    Color::TRANSPARENT,
                    Modifiers::NONE,
                );
            }
        }

        // Left and right borders
        for cy in y..y.saturating_add(h).min(self.buffer.height()) {
            self.buffer
                .update(x, cy, "│", color, Color::TRANSPARENT, Modifiers::NONE);
            if w > 0 {
                self.buffer.update(
                    x.saturating_add(w - 1).min(self.buffer.width() - 1),
                    cy,
                    "│",
                    color,
                    Color::TRANSPARENT,
                    Modifiers::NONE,
                );
            }
        }
    }

    fn draw_text(&mut self, text: &str, position: Point, style: &TextStyle) {
        let x = position.x.max(0.0) as u16;
        let y = position.y.max(0.0) as u16;

        if y >= self.buffer.height() {
            return;
        }

        let modifiers = if style.weight == FontWeight::Bold {
            Modifiers::BOLD
        } else {
            Modifiers::NONE
        };

        let mut cx = x;
        for ch in text.chars() {
            if cx >= self.buffer.width() {
                break;
            }
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            self.buffer
                .update(cx, y, s, style.color, Color::TRANSPARENT, modifiers);
            cx = cx.saturating_add(1);
        }
    }

    fn draw_line(&mut self, from: Point, to: Point, color: Color, _width: f32) {
        // Simple Bresenham line for terminal
        let x0 = from.x as i32;
        let y0 = from.y as i32;
        let x1 = to.x as i32;
        let y1 = to.y as i32;

        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        let mut x = x0;
        let mut y = y0;

        loop {
            if x >= 0
                && y >= 0
                && (x as u16) < self.buffer.width()
                && (y as u16) < self.buffer.height()
            {
                self.buffer.update(
                    x as u16,
                    y as u16,
                    "•",
                    color,
                    Color::TRANSPARENT,
                    Modifiers::NONE,
                );
            }

            if x == x1 && y == y1 {
                break;
            }

            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    fn fill_circle(&mut self, center: Point, radius: f32, color: Color) {
        let cx = center.x as i32;
        let cy = center.y as i32;
        let r = radius as i32;

        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    let x = cx + dx;
                    let y = cy + dy;
                    if x >= 0
                        && y >= 0
                        && (x as u16) < self.buffer.width()
                        && (y as u16) < self.buffer.height()
                    {
                        self.buffer.update(
                            x as u16,
                            y as u16,
                            "●",
                            color,
                            Color::TRANSPARENT,
                            Modifiers::NONE,
                        );
                    }
                }
            }
        }
    }

    fn stroke_circle(&mut self, center: Point, radius: f32, color: Color, _width: f32) {
        let cx = center.x as i32;
        let cy = center.y as i32;
        let r = radius as i32;

        // Simple circle approximation
        for i in 0..360 {
            let angle = (i as f32).to_radians();
            let x = cx + (r as f32 * angle.cos()) as i32;
            let y = cy + (r as f32 * angle.sin()) as i32;
            if x >= 0
                && y >= 0
                && (x as u16) < self.buffer.width()
                && (y as u16) < self.buffer.height()
            {
                self.buffer.update(
                    x as u16,
                    y as u16,
                    "○",
                    color,
                    Color::TRANSPARENT,
                    Modifiers::NONE,
                );
            }
        }
    }

    fn fill_arc(&mut self, _center: Point, _radius: f32, _start: f32, _end: f32, _color: Color) {
        // Arc rendering not needed for benchmarking
    }

    fn draw_path(&mut self, points: &[Point], color: Color, width: f32) {
        for window in points.windows(2) {
            self.draw_line(window[0], window[1], color, width);
        }
    }

    fn fill_polygon(&mut self, _points: &[Point], _color: Color) {
        // Polygon fill not needed for benchmarking
    }

    fn push_clip(&mut self, _rect: Rect) {
        // Clipping not implemented for headless canvas
    }

    fn pop_clip(&mut self) {
        // Clipping not implemented for headless canvas
    }

    fn push_transform(&mut self, _transform: Transform2D) {
        // Transforms not implemented for headless canvas
    }

    fn pop_transform(&mut self) {
        // Transforms not implemented for headless canvas
    }
}

// ============================================================================
// RenderMetrics
// ============================================================================

/// Performance metrics collected during rendering.
#[derive(Debug, Clone)]
pub struct RenderMetrics {
    /// Total frames rendered.
    pub frame_count: u64,
    /// Frame time statistics.
    pub frame_times: FrameTimeStats,
    /// Memory statistics.
    pub memory: MemoryStats,
    /// Widget-level breakdown.
    pub widget_times: HashMap<String, FrameTimeStats>,
}

impl RenderMetrics {
    /// Create new metrics collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            frame_count: 0,
            frame_times: FrameTimeStats::new(),
            memory: MemoryStats::default(),
            widget_times: HashMap::new(),
        }
    }

    /// Record a frame's render time.
    pub fn record_frame(&mut self, duration: Duration) {
        self.frame_count += 1;
        self.frame_times.record(duration);
    }

    /// Record widget-specific timing.
    pub fn record_widget(&mut self, name: &str, duration: Duration) {
        self.widget_times
            .entry(name.to_string())
            .or_default()
            .record(duration);
    }

    /// Check if metrics meet performance targets.
    #[must_use]
    pub fn meets_targets(&self, targets: &PerformanceTargets) -> bool {
        self.frame_times.max_us <= targets.max_frame_us
            && self.frame_times.p99_us <= targets.p99_frame_us
            && self.memory.steady_state_bytes <= targets.max_memory_bytes
            && self.memory.allocations_per_frame <= targets.max_allocs_per_frame
    }

    /// Export to JSON.
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            r#"{{
  "frame_count": {},
  "frame_times": {{
    "min_us": {},
    "max_us": {},
    "mean_us": {:.1},
    "p50_us": {},
    "p95_us": {},
    "p99_us": {},
    "stddev_us": {:.1}
  }},
  "memory": {{
    "peak_bytes": {},
    "steady_state_bytes": {},
    "allocations_per_frame": {:.2}
  }}
}}"#,
            self.frame_count,
            self.frame_times.min_us,
            self.frame_times.max_us,
            self.frame_times.mean_us,
            self.frame_times.p50_us,
            self.frame_times.p95_us,
            self.frame_times.p99_us,
            self.frame_times.stddev_us,
            self.memory.peak_bytes,
            self.memory.steady_state_bytes,
            self.memory.allocations_per_frame,
        )
    }

    /// Export to CSV row.
    #[must_use]
    pub fn to_csv_row(&self, widget_name: &str, width: u16, height: u16) -> String {
        format!(
            "{},{},{},{},{},{},{:.1},{},{},{},{}",
            widget_name,
            width,
            height,
            self.frame_count,
            self.frame_times.min_us,
            self.frame_times.max_us,
            self.frame_times.mean_us,
            self.frame_times.p50_us,
            self.frame_times.p95_us,
            self.frame_times.p99_us,
            self.memory.steady_state_bytes,
        )
    }

    /// Get CSV header.
    #[must_use]
    pub fn csv_header() -> &'static str {
        "widget,width,height,frames,min_us,max_us,mean_us,p50_us,p95_us,p99_us,memory_bytes"
    }
}

impl Default for RenderMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Frame time statistics with percentiles.
#[derive(Debug, Clone, Default)]
pub struct FrameTimeStats {
    /// Minimum frame time in microseconds.
    pub min_us: u64,
    /// Maximum frame time in microseconds.
    pub max_us: u64,
    /// Mean frame time in microseconds.
    pub mean_us: f64,
    /// 50th percentile (median) frame time.
    pub p50_us: u64,
    /// 95th percentile frame time.
    pub p95_us: u64,
    /// 99th percentile frame time.
    pub p99_us: u64,
    /// Standard deviation in microseconds.
    pub stddev_us: f64,
    /// Raw samples (for percentile calculation).
    samples: Vec<u64>,
}

impl FrameTimeStats {
    /// Create new frame time stats.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_us: u64::MAX,
            max_us: 0,
            mean_us: 0.0,
            p50_us: 0,
            p95_us: 0,
            p99_us: 0,
            stddev_us: 0.0,
            samples: Vec::with_capacity(1024),
        }
    }

    /// Record a frame time sample.
    pub fn record(&mut self, duration: Duration) {
        let us = duration.as_micros() as u64;
        self.samples.push(us);

        self.min_us = self.min_us.min(us);
        self.max_us = self.max_us.max(us);

        // Update running mean
        let n = self.samples.len() as f64;
        self.mean_us = self.mean_us + (us as f64 - self.mean_us) / n;
    }

    /// Finalize statistics (calculate percentiles and stddev).
    pub fn finalize(&mut self) {
        if self.samples.is_empty() {
            return;
        }

        // Sort for percentile calculation
        self.samples.sort_unstable();

        let n = self.samples.len();
        self.p50_us = self.samples[n / 2];
        self.p95_us = self.samples[(n as f64 * 0.95) as usize];
        self.p99_us = self.samples[(n as f64 * 0.99).min((n - 1) as f64) as usize];

        // Calculate standard deviation
        let variance: f64 = self
            .samples
            .iter()
            .map(|&x| {
                let diff = x as f64 - self.mean_us;
                diff * diff
            })
            .sum::<f64>()
            / n as f64;
        self.stddev_us = variance.sqrt();
    }

    /// Get sample count.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

/// Memory statistics.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Peak memory usage in bytes.
    pub peak_bytes: usize,
    /// Steady-state memory usage in bytes.
    pub steady_state_bytes: usize,
    /// Average allocations per frame.
    pub allocations_per_frame: f64,
}

// ============================================================================
// PerformanceTargets
// ============================================================================

/// Performance targets for validation.
#[derive(Debug, Clone)]
pub struct PerformanceTargets {
    /// Maximum frame time in microseconds.
    pub max_frame_us: u64,
    /// Target p99 frame time.
    pub p99_frame_us: u64,
    /// Maximum memory usage.
    pub max_memory_bytes: usize,
    /// Maximum allocations per frame.
    pub max_allocs_per_frame: f64,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            max_frame_us: 16_667,         // 60fps = 16.67ms
            p99_frame_us: 1_000,          // 1ms for TUI
            max_memory_bytes: 100 * 1024, // 100KB
            max_allocs_per_frame: 0.0,    // Zero-allocation target
        }
    }
}

impl PerformanceTargets {
    /// Create targets for 60fps rendering.
    #[must_use]
    pub fn for_60fps() -> Self {
        Self::default()
    }

    /// Create targets for 30fps rendering.
    #[must_use]
    pub fn for_30fps() -> Self {
        Self {
            max_frame_us: 33_333,
            p99_frame_us: 5_000,
            ..Self::default()
        }
    }

    /// Create strict targets for high-performance scenarios.
    #[must_use]
    pub fn strict() -> Self {
        Self {
            max_frame_us: 1_000, // 1ms max
            p99_frame_us: 500,   // 500us p99
            max_memory_bytes: 50 * 1024,
            max_allocs_per_frame: 0.0,
        }
    }
}

// ============================================================================
// BenchmarkHarness
// ============================================================================

/// Harness for running widget benchmarks.
#[derive(Debug)]
pub struct BenchmarkHarness {
    /// Headless canvas for rendering.
    canvas: HeadlessCanvas,
    /// Number of warmup frames.
    warmup_frames: u32,
    /// Number of benchmark frames.
    benchmark_frames: u32,
    /// Deterministic mode.
    deterministic: bool,
}

impl BenchmarkHarness {
    /// Create new benchmark harness with given dimensions.
    #[must_use]
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            canvas: HeadlessCanvas::new(width, height),
            warmup_frames: 100,
            benchmark_frames: 1000,
            deterministic: true,
        }
    }

    /// Set warmup and benchmark frame counts.
    #[must_use]
    pub fn with_frames(mut self, warmup: u32, benchmark: u32) -> Self {
        self.warmup_frames = warmup;
        self.benchmark_frames = benchmark;
        self
    }

    /// Enable/disable deterministic mode.
    #[must_use]
    pub fn with_deterministic(mut self, deterministic: bool) -> Self {
        self.deterministic = deterministic;
        self.canvas = self.canvas.with_deterministic(deterministic);
        self
    }

    /// Run benchmark on a widget.
    pub fn benchmark<W: Widget>(&mut self, widget: &mut W, bounds: Rect) -> BenchmarkResult {
        // Warmup phase
        for _ in 0..self.warmup_frames {
            self.canvas.clear();
            widget.layout(bounds);
            widget.paint(&mut self.canvas);
        }

        // Reset metrics
        self.canvas.reset_metrics();

        // Benchmark phase
        for _ in 0..self.benchmark_frames {
            let start = Instant::now();
            self.canvas.clear();
            widget.layout(bounds);
            widget.paint(&mut self.canvas);
            let elapsed = start.elapsed();
            self.canvas.metrics_mut().record_frame(elapsed);
        }

        // Finalize statistics
        self.canvas.metrics_mut().frame_times.finalize();

        BenchmarkResult {
            widget_name: widget.brick_name().to_string(),
            metrics: self.canvas.metrics().clone(),
            final_frame: self.canvas.dump(),
            width: self.canvas.width(),
            height: self.canvas.height(),
        }
    }

    /// Run comparison benchmark between two widgets.
    pub fn compare<W1: Widget, W2: Widget>(
        &mut self,
        widget_a: &mut W1,
        widget_b: &mut W2,
        bounds: Rect,
    ) -> ComparisonResult {
        let result_a = self.benchmark(widget_a, bounds);

        // Reset canvas for second widget
        self.canvas = HeadlessCanvas::new(self.canvas.width(), self.canvas.height())
            .with_deterministic(self.deterministic);

        let result_b = self.benchmark(widget_b, bounds);

        ComparisonResult {
            widget_a: result_a,
            widget_b: result_b,
        }
    }

    /// Get reference to canvas.
    #[must_use]
    pub fn canvas(&self) -> &HeadlessCanvas {
        &self.canvas
    }

    /// Get mutable reference to canvas.
    pub fn canvas_mut(&mut self) -> &mut HeadlessCanvas {
        &mut self.canvas
    }
}

/// Result from a single widget benchmark.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the widget.
    pub widget_name: String,
    /// Collected metrics.
    pub metrics: RenderMetrics,
    /// Final frame output (for snapshot comparison).
    pub final_frame: String,
    /// Canvas width.
    pub width: u16,
    /// Canvas height.
    pub height: u16,
}

impl BenchmarkResult {
    /// Check if result meets performance targets.
    #[must_use]
    pub fn meets_targets(&self, targets: &PerformanceTargets) -> bool {
        self.metrics.meets_targets(targets)
    }

    /// Export to JSON.
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            r#"{{
  "widget": "{}",
  "dimensions": {{ "width": {}, "height": {} }},
  "metrics": {},
  "meets_targets": {}
}}"#,
            self.widget_name,
            self.width,
            self.height,
            self.metrics.to_json(),
            self.metrics.meets_targets(&PerformanceTargets::default()),
        )
    }
}

/// Result from comparing two widgets.
#[derive(Debug)]
pub struct ComparisonResult {
    /// First widget result.
    pub widget_a: BenchmarkResult,
    /// Second widget result.
    pub widget_b: BenchmarkResult,
}

impl ComparisonResult {
    /// Check if `widget_a` is faster than `widget_b`.
    #[must_use]
    pub fn a_is_faster(&self) -> bool {
        self.widget_a.metrics.frame_times.mean_us < self.widget_b.metrics.frame_times.mean_us
    }

    /// Get speedup ratio (b/a).
    #[must_use]
    pub fn speedup_ratio(&self) -> f64 {
        if self.widget_a.metrics.frame_times.mean_us > 0.0 {
            self.widget_b.metrics.frame_times.mean_us / self.widget_a.metrics.frame_times.mean_us
        } else {
            1.0
        }
    }

    /// Get performance summary.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} mean: {:.1}us, {} mean: {:.1}us, speedup: {:.2}x",
            self.widget_a.widget_name,
            self.widget_a.metrics.frame_times.mean_us,
            self.widget_b.widget_name,
            self.widget_b.metrics.frame_times.mean_us,
            self.speedup_ratio(),
        )
    }
}

// ============================================================================
// DeterministicContext
// ============================================================================

/// Deterministic rendering context for reproducible benchmarks.
///
/// Provides fixed timestamps, RNG seeds, and simulated system data
/// for pixel-perfect comparison testing.
#[derive(Debug, Clone)]
pub struct DeterministicContext {
    /// Fixed timestamp (epoch seconds).
    pub timestamp: u64,
    /// Fixed RNG seed.
    pub rng_seed: u64,
    /// Current RNG state.
    rng_state: u64,
    /// Simulated CPU usage per core.
    pub cpu_usage: Vec<f64>,
    /// Simulated memory usage (bytes).
    pub memory_used: u64,
    /// Simulated memory total (bytes).
    pub memory_total: u64,
}

impl DeterministicContext {
    /// Create deterministic context with default values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            // Fixed to 2026-01-01 00:00:00 UTC
            timestamp: 1_767_225_600,
            rng_seed: 42,
            rng_state: 42,
            cpu_usage: vec![45.0, 32.0, 67.0, 12.0, 89.0, 23.0, 56.0, 78.0],
            memory_used: 18_200_000_000,  // 18.2 GB
            memory_total: 32_000_000_000, // 32 GB
        }
    }

    /// Create with custom seed for reproducible random data.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng_seed: seed,
            rng_state: seed,
            ..Self::new()
        }
    }

    /// Get deterministic timestamp.
    #[must_use]
    pub const fn now(&self) -> u64 {
        self.timestamp
    }

    /// Get reproducible random value (0.0-1.0).
    pub fn rand(&mut self) -> f64 {
        // Simple xorshift64
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    /// Get reproducible random value in range.
    pub fn rand_range(&mut self, min: f64, max: f64) -> f64 {
        min + self.rand() * (max - min)
    }

    /// Get CPU usage for a specific core.
    #[must_use]
    pub fn get_cpu_usage(&self, core: usize) -> f64 {
        self.cpu_usage.get(core).copied().unwrap_or(0.0)
    }

    /// Get memory usage percentage.
    #[must_use]
    pub fn memory_percent(&self) -> f64 {
        if self.memory_total > 0 {
            (self.memory_used as f64 / self.memory_total as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Reset RNG to initial state.
    pub fn reset_rng(&mut self) {
        self.rng_state = self.rng_seed;
    }
}

impl Default for DeterministicContext {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use presentar_core::{
        Brick, BrickAssertion, BrickBudget, BrickVerification, Constraints, Event, LayoutResult,
        Size, TypeId,
    };
    use std::any::Any;

    // Simple test widget
    #[derive(Debug)]
    struct TestWidget {
        bounds: Rect,
    }

    impl TestWidget {
        fn new() -> Self {
            Self {
                bounds: Rect::default(),
            }
        }
    }

    impl Brick for TestWidget {
        fn brick_name(&self) -> &'static str {
            "test_widget"
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &[]
        }

        fn budget(&self) -> BrickBudget {
            BrickBudget::uniform(1)
        }

        fn verify(&self) -> BrickVerification {
            BrickVerification {
                passed: vec![],
                failed: vec![],
                verification_time: Duration::from_micros(1),
            }
        }

        fn to_html(&self) -> String {
            String::new()
        }

        fn to_css(&self) -> String {
            String::new()
        }
    }

    impl Widget for TestWidget {
        fn type_id(&self) -> TypeId {
            TypeId::of::<Self>()
        }

        fn measure(&self, constraints: Constraints) -> Size {
            constraints.constrain(Size::new(10.0, 5.0))
        }

        fn layout(&mut self, bounds: Rect) -> LayoutResult {
            self.bounds = bounds;
            LayoutResult {
                size: Size::new(bounds.width, bounds.height),
            }
        }

        fn paint(&self, canvas: &mut dyn Canvas) {
            canvas.fill_rect(self.bounds, Color::BLUE);
            canvas.draw_text(
                "Test",
                Point::new(self.bounds.x, self.bounds.y),
                &TextStyle::default(),
            );
        }

        fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
            None
        }

        fn children(&self) -> &[Box<dyn Widget>] {
            &[]
        }

        fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
            &mut []
        }
    }

    #[test]
    fn test_headless_canvas_new() {
        let canvas = HeadlessCanvas::new(80, 24);
        assert_eq!(canvas.width(), 80);
        assert_eq!(canvas.height(), 24);
        assert_eq!(canvas.frame_count(), 0);
    }

    #[test]
    fn test_headless_canvas_deterministic() {
        let canvas = HeadlessCanvas::new(80, 24).with_deterministic(true);
        assert!(canvas.is_deterministic());
    }

    #[test]
    fn test_headless_canvas_render_frame() {
        let mut canvas = HeadlessCanvas::new(80, 24);
        canvas.render_frame(|c| {
            c.draw_text("Hello", Point::new(0.0, 0.0), &TextStyle::default());
        });
        assert_eq!(canvas.frame_count(), 1);
        assert!(canvas.metrics().frame_times.sample_count() > 0);
    }

    #[test]
    fn test_headless_canvas_dump() {
        let mut canvas = HeadlessCanvas::new(10, 2);
        canvas.draw_text("Hi", Point::new(0.0, 0.0), &TextStyle::default());
        let dump = canvas.dump();
        assert!(dump.contains("Hi"));
    }

    #[test]
    fn test_headless_canvas_clear() {
        let mut canvas = HeadlessCanvas::new(10, 10);
        canvas.draw_text("Test", Point::new(0.0, 0.0), &TextStyle::default());
        canvas.clear();
        // After clear, buffer should be reset
        assert_eq!(canvas.buffer().dirty_count(), 100); // All cells dirty after clear
    }

    #[test]
    fn test_headless_canvas_fill_rect() {
        let mut canvas = HeadlessCanvas::new(20, 10);
        canvas.fill_rect(Rect::new(5.0, 2.0, 3.0, 3.0), Color::RED);
        // Check that cells were updated
        let cell = canvas.buffer().get(6, 3).unwrap();
        assert_eq!(cell.bg, Color::RED);
    }

    #[test]
    fn test_headless_canvas_draw_line() {
        let mut canvas = HeadlessCanvas::new(20, 10);
        canvas.draw_line(
            Point::new(0.0, 0.0),
            Point::new(5.0, 5.0),
            Color::GREEN,
            1.0,
        );
        // Line should have been drawn
        let cell = canvas.buffer().get(0, 0).unwrap();
        assert_eq!(cell.fg, Color::GREEN);
    }

    #[test]
    fn test_render_metrics_new() {
        let metrics = RenderMetrics::new();
        assert_eq!(metrics.frame_count, 0);
        assert_eq!(metrics.frame_times.sample_count(), 0);
    }

    #[test]
    fn test_render_metrics_record_frame() {
        let mut metrics = RenderMetrics::new();
        metrics.record_frame(Duration::from_micros(100));
        metrics.record_frame(Duration::from_micros(200));
        assert_eq!(metrics.frame_count, 2);
        assert_eq!(metrics.frame_times.sample_count(), 2);
    }

    #[test]
    fn test_render_metrics_meets_targets() {
        let mut metrics = RenderMetrics::new();
        metrics.record_frame(Duration::from_micros(500));
        metrics.frame_times.finalize();

        let targets = PerformanceTargets::default();
        assert!(metrics.meets_targets(&targets));
    }

    #[test]
    fn test_render_metrics_to_json() {
        let mut metrics = RenderMetrics::new();
        metrics.record_frame(Duration::from_micros(100));
        metrics.frame_times.finalize();

        let json = metrics.to_json();
        assert!(json.contains("frame_count"));
        assert!(json.contains("frame_times"));
    }

    #[test]
    fn test_frame_time_stats_finalize() {
        let mut stats = FrameTimeStats::new();
        for i in 0..100 {
            stats.record(Duration::from_micros(100 + i));
        }
        stats.finalize();

        assert!(stats.min_us >= 100);
        assert!(stats.max_us <= 199);
        assert!(stats.p50_us > 0);
        assert!(stats.p95_us > 0);
        assert!(stats.p99_us > 0);
    }

    #[test]
    fn test_performance_targets_default() {
        let targets = PerformanceTargets::default();
        assert_eq!(targets.max_frame_us, 16_667);
        assert_eq!(targets.p99_frame_us, 1_000);
    }

    #[test]
    fn test_performance_targets_strict() {
        let targets = PerformanceTargets::strict();
        assert_eq!(targets.max_frame_us, 1_000);
        assert_eq!(targets.p99_frame_us, 500);
    }

    #[test]
    fn test_benchmark_harness_new() {
        let harness = BenchmarkHarness::new(80, 24);
        assert_eq!(harness.canvas().width(), 80);
        assert_eq!(harness.canvas().height(), 24);
    }

    #[test]
    fn test_benchmark_harness_with_frames() {
        let harness = BenchmarkHarness::new(80, 24).with_frames(10, 100);
        assert_eq!(harness.warmup_frames, 10);
        assert_eq!(harness.benchmark_frames, 100);
    }

    #[test]
    fn test_benchmark_harness_benchmark() {
        let mut harness = BenchmarkHarness::new(40, 10).with_frames(5, 20);
        let mut widget = TestWidget::new();
        let bounds = Rect::new(0.0, 0.0, 40.0, 10.0);

        let result = harness.benchmark(&mut widget, bounds);

        assert_eq!(result.widget_name, "test_widget");
        assert_eq!(result.metrics.frame_count, 20);
        assert!(!result.final_frame.is_empty());
    }

    #[test]
    fn test_benchmark_harness_compare() {
        let mut harness = BenchmarkHarness::new(40, 10).with_frames(5, 10);
        let mut widget_a = TestWidget::new();
        let mut widget_b = TestWidget::new();
        let bounds = Rect::new(0.0, 0.0, 40.0, 10.0);

        let result = harness.compare(&mut widget_a, &mut widget_b, bounds);

        assert_eq!(result.widget_a.widget_name, "test_widget");
        assert_eq!(result.widget_b.widget_name, "test_widget");
        assert!(result.speedup_ratio() > 0.0);
    }

    #[test]
    fn test_benchmark_result_to_json() {
        let result = BenchmarkResult {
            widget_name: "test".to_string(),
            metrics: RenderMetrics::new(),
            final_frame: "frame".to_string(),
            width: 80,
            height: 24,
        };

        let json = result.to_json();
        assert!(json.contains("test"));
        assert!(json.contains("80"));
    }

    #[test]
    fn test_comparison_result_summary() {
        let result_a = BenchmarkResult {
            widget_name: "widget_a".to_string(),
            metrics: RenderMetrics::new(),
            final_frame: String::new(),
            width: 80,
            height: 24,
        };
        let result_b = BenchmarkResult {
            widget_name: "widget_b".to_string(),
            metrics: RenderMetrics::new(),
            final_frame: String::new(),
            width: 80,
            height: 24,
        };

        let comparison = ComparisonResult {
            widget_a: result_a,
            widget_b: result_b,
        };

        let summary = comparison.summary();
        assert!(summary.contains("widget_a"));
        assert!(summary.contains("widget_b"));
    }

    #[test]
    fn test_deterministic_context_new() {
        let ctx = DeterministicContext::new();
        assert_eq!(ctx.timestamp, 1767225600);
        assert_eq!(ctx.rng_seed, 42);
        assert_eq!(ctx.cpu_usage.len(), 8);
    }

    #[test]
    fn test_deterministic_context_with_seed() {
        let ctx = DeterministicContext::with_seed(123);
        assert_eq!(ctx.rng_seed, 123);
    }

    #[test]
    fn test_deterministic_context_rand() {
        let mut ctx = DeterministicContext::new();
        let r1 = ctx.rand();
        let r2 = ctx.rand();
        assert!((0.0..=1.0).contains(&r1));
        assert!((0.0..=1.0).contains(&r2));
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_deterministic_context_rand_reproducible() {
        let mut ctx1 = DeterministicContext::with_seed(42);
        let mut ctx2 = DeterministicContext::with_seed(42);

        let r1 = ctx1.rand();
        let r2 = ctx2.rand();
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_deterministic_context_rand_range() {
        let mut ctx = DeterministicContext::new();
        let r = ctx.rand_range(10.0, 20.0);
        assert!((10.0..=20.0).contains(&r));
    }

    #[test]
    fn test_deterministic_context_reset_rng() {
        let mut ctx = DeterministicContext::new();
        let r1 = ctx.rand();
        ctx.reset_rng();
        let r2 = ctx.rand();
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_deterministic_context_get_cpu_usage() {
        let ctx = DeterministicContext::new();
        assert_eq!(ctx.get_cpu_usage(0), 45.0);
        assert_eq!(ctx.get_cpu_usage(7), 78.0);
        assert_eq!(ctx.get_cpu_usage(100), 0.0); // Out of bounds
    }

    #[test]
    fn test_deterministic_context_memory_percent() {
        let ctx = DeterministicContext::new();
        let percent = ctx.memory_percent();
        assert!(percent > 50.0 && percent < 60.0); // ~56.875%
    }

    #[test]
    fn test_render_metrics_record_widget() {
        let mut metrics = RenderMetrics::new();
        metrics.record_widget("cpu_grid", Duration::from_micros(100));
        metrics.record_widget("cpu_grid", Duration::from_micros(150));
        metrics.record_widget("memory_bar", Duration::from_micros(50));

        assert!(metrics.widget_times.contains_key("cpu_grid"));
        assert!(metrics.widget_times.contains_key("memory_bar"));
        assert_eq!(metrics.widget_times["cpu_grid"].sample_count(), 2);
    }

    #[test]
    fn test_render_metrics_csv_row() {
        let mut metrics = RenderMetrics::new();
        metrics.record_frame(Duration::from_micros(100));
        metrics.frame_times.finalize();

        let csv = metrics.to_csv_row("test_widget", 80, 24);
        assert!(csv.contains("test_widget"));
        assert!(csv.contains("80"));
        assert!(csv.contains("24"));
    }

    #[test]
    fn test_render_metrics_csv_header() {
        let header = RenderMetrics::csv_header();
        assert!(header.contains("widget"));
        assert!(header.contains("min_us"));
        assert!(header.contains("p99_us"));
    }

    #[test]
    fn test_headless_canvas_reset_metrics() {
        let mut canvas = HeadlessCanvas::new(80, 24);
        canvas.render_frame(|_| {});
        canvas.render_frame(|_| {});
        assert_eq!(canvas.frame_count(), 2);

        canvas.reset_metrics();
        assert_eq!(canvas.frame_count(), 0);
        assert_eq!(canvas.metrics().frame_count, 0);
    }

    #[test]
    fn test_benchmark_result_meets_targets() {
        let mut metrics = RenderMetrics::new();
        metrics.record_frame(Duration::from_micros(500));
        metrics.frame_times.finalize();

        let result = BenchmarkResult {
            widget_name: "test".to_string(),
            metrics,
            final_frame: String::new(),
            width: 80,
            height: 24,
        };

        assert!(result.meets_targets(&PerformanceTargets::default()));
    }

    #[test]
    fn test_comparison_result_a_is_faster() {
        let mut metrics_a = RenderMetrics::new();
        metrics_a.frame_times.mean_us = 100.0;

        let mut metrics_b = RenderMetrics::new();
        metrics_b.frame_times.mean_us = 200.0;

        let comparison = ComparisonResult {
            widget_a: BenchmarkResult {
                widget_name: "a".to_string(),
                metrics: metrics_a,
                final_frame: String::new(),
                width: 80,
                height: 24,
            },
            widget_b: BenchmarkResult {
                widget_name: "b".to_string(),
                metrics: metrics_b,
                final_frame: String::new(),
                width: 80,
                height: 24,
            },
        };

        assert!(comparison.a_is_faster());
        assert!((comparison.speedup_ratio() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_frame_time_stats_empty() {
        let mut stats = FrameTimeStats::new();
        stats.finalize();
        // Should not panic with empty samples
        assert_eq!(stats.sample_count(), 0);
    }

    #[test]
    fn test_performance_targets_for_30fps() {
        let targets = PerformanceTargets::for_30fps();
        assert_eq!(targets.max_frame_us, 33_333);
    }

    #[test]
    fn test_headless_canvas_stroke_rect() {
        let mut canvas = HeadlessCanvas::new(20, 10);
        canvas.stroke_rect(Rect::new(2.0, 2.0, 5.0, 3.0), Color::RED, 1.0);
        // Top border should have horizontal line char
        let cell = canvas.buffer().get(3, 2).unwrap();
        assert_eq!(cell.symbol.as_str(), "─");
    }

    #[test]
    fn test_headless_canvas_fill_circle() {
        let mut canvas = HeadlessCanvas::new(20, 20);
        canvas.fill_circle(Point::new(10.0, 10.0), 3.0, Color::GREEN);
        // Center should be filled
        let cell = canvas.buffer().get(10, 10).unwrap();
        assert_eq!(cell.fg, Color::GREEN);
    }

    #[test]
    fn test_headless_canvas_draw_path() {
        let mut canvas = HeadlessCanvas::new(20, 10);
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(5.0, 0.0),
            Point::new(5.0, 5.0),
        ];
        canvas.draw_path(&points, Color::BLUE, 1.0);
        // Path should have been drawn
        let cell = canvas.buffer().get(2, 0).unwrap();
        assert_eq!(cell.fg, Color::BLUE);
    }

    #[test]
    fn test_headless_canvas_buffer_mut() {
        let mut canvas = HeadlessCanvas::new(20, 10);
        let buffer = canvas.buffer_mut();
        buffer.update(0, 0, "X", Color::RED, Color::TRANSPARENT, Modifiers::NONE);
        let cell = canvas.buffer().get(0, 0).unwrap();
        assert_eq!(cell.symbol.as_str(), "X");
    }

    #[test]
    fn test_headless_canvas_stroke_circle() {
        let mut canvas = HeadlessCanvas::new(30, 30);
        canvas.stroke_circle(Point::new(15.0, 15.0), 5.0, Color::RED, 1.0);
        // Some points on the circle perimeter should be filled
        // Since it's drawn with 360 iterations, there should be some marks
    }

    #[test]
    fn test_headless_canvas_fill_arc() {
        let mut canvas = HeadlessCanvas::new(20, 20);
        // fill_arc is a no-op for headless canvas, but should not panic
        canvas.fill_arc(
            Point::new(10.0, 10.0),
            5.0,
            0.0,
            std::f32::consts::PI,
            Color::GREEN,
        );
    }

    #[test]
    fn test_headless_canvas_fill_polygon() {
        let mut canvas = HeadlessCanvas::new(20, 20);
        // fill_polygon is a no-op for headless canvas, but should not panic
        canvas.fill_polygon(
            &[
                Point::new(0.0, 0.0),
                Point::new(10.0, 0.0),
                Point::new(5.0, 10.0),
            ],
            Color::BLUE,
        );
    }

    #[test]
    fn test_deterministic_context_now() {
        let ctx = DeterministicContext::new();
        // Timestamp should be 2026-01-01 00:00:00 UTC
        assert_eq!(ctx.now(), 1767225600);
    }

    #[test]
    fn test_deterministic_context_default() {
        let ctx = DeterministicContext::default();
        assert_eq!(ctx.timestamp, 1767225600);
    }

    #[test]
    fn test_deterministic_context_memory_percent_zero_total() {
        let ctx = DeterministicContext {
            timestamp: 0,
            rng_seed: 42,
            rng_state: 42,
            cpu_usage: vec![],
            memory_used: 100,
            memory_total: 0, // Division by zero case
        };
        assert_eq!(ctx.memory_percent(), 0.0);
    }

    #[test]
    fn test_test_widget_brick_traits() {
        let widget = TestWidget::new();
        assert_eq!(widget.brick_name(), "test_widget");
        assert!(widget.assertions().is_empty());
        assert_eq!(widget.budget().total_ms, 1);
        assert!(widget.verify().passed.is_empty());
        assert!(widget.to_html().is_empty());
        assert!(widget.to_css().is_empty());
    }

    #[test]
    fn test_test_widget_widget_traits() {
        use presentar_core::Widget;
        let mut widget = TestWidget::new();
        let _ = Widget::type_id(&widget);
        let size = widget.measure(Constraints::new(0.0, 100.0, 0.0, 100.0));
        assert_eq!(size.width, 10.0);
        assert_eq!(size.height, 5.0);

        widget.layout(Rect::new(0.0, 0.0, 20.0, 10.0));
        assert_eq!(widget.bounds.width, 20.0);

        // Test event returns None
        let result = widget.event(&Event::Resize {
            width: 80.0,
            height: 24.0,
        });
        assert!(result.is_none());

        assert!(widget.children().is_empty());
        assert!(widget.children_mut().is_empty());
    }
}
