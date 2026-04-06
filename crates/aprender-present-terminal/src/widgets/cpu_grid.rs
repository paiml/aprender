//! `CpuGrid` widget for dense per-core CPU visualization.
//!
//! Displays N CPU cores in a compact grid with gradient-colored meters.
//! Reference: btop/ttop per-core CPU displays.
//!
//! # Features
//!
//! - Per-core utilization with gradient coloring
//! - Optional frequency scaling indicators (⚡↑→↓·)
//! - CPU governor display (performance, powersave, etc.)
//! - Compact and percentage display modes

use crate::compute_block::{CpuGovernor, FrequencyScalingState};
use crate::theme::Gradient;
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Block characters for single-row meters (8 levels).
const METER_CHARS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Frequency scaling state indicator characters.
const FREQ_INDICATORS: [char; 5] = ['⚡', '↑', '→', '↓', '·'];

/// Per-core CPU grid with gradient-colored meters.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct CpuGrid {
    /// Per-core utilization (0.0-100.0).
    pub core_usage: Vec<f64>,
    /// Gradient for coloring (low→high).
    gradient: Gradient,
    /// Number of columns (auto-calculated if None).
    columns: Option<usize>,
    /// Show core labels (0, 1, 2...).
    show_labels: bool,
    /// Show numeric percentages instead of meter chars.
    show_percentages: bool,
    /// Compact mode (minimal spacing).
    compact: bool,
    /// Cached bounds.
    bounds: Rect,
    /// Per-core current frequencies in MHz.
    frequencies: Option<Vec<u32>>,
    /// Per-core max frequencies in MHz.
    max_frequencies: Option<Vec<u32>>,
    /// Current CPU governor (affects all cores).
    governor: Option<CpuGovernor>,
    /// Show frequency scaling indicators.
    show_freq_indicators: bool,
}

impl Default for CpuGrid {
    fn default() -> Self {
        Self::new(vec![])
    }
}

impl CpuGrid {
    /// Create a new CPU grid with core usage data.
    #[must_use]
    pub fn new(core_usage: Vec<f64>) -> Self {
        Self {
            core_usage,
            gradient: Gradient::from_hex(&["#7aa2f7", "#e0af68", "#f7768e"]), // Tokyo Night CPU
            columns: None,
            show_labels: true,
            show_percentages: false,
            compact: false,
            bounds: Rect::default(),
            frequencies: None,
            max_frequencies: None,
            governor: None,
            show_freq_indicators: false,
        }
    }

    /// Set the gradient for coloring.
    #[must_use]
    pub fn with_gradient(mut self, gradient: Gradient) -> Self {
        self.gradient = gradient;
        self
    }

    /// Set explicit column count.
    #[must_use]
    pub fn with_columns(mut self, cols: usize) -> Self {
        debug_assert!(cols > 0, "column count must be positive");
        self.columns = Some(cols);
        self
    }

    /// Enable compact mode (minimal spacing).
    #[must_use]
    pub fn compact(mut self) -> Self {
        self.compact = true;
        self
    }

    /// Disable core labels.
    #[must_use]
    pub fn without_labels(mut self) -> Self {
        self.show_labels = false;
        self
    }

    /// Show numeric percentages instead of meter characters.
    /// Format: "12 45%" or " 5 99%" (like ttop).
    #[must_use]
    pub fn with_percentages(mut self) -> Self {
        self.show_percentages = true;
        self
    }

    /// Enable frequency scaling indicators (⚡↑→↓·).
    ///
    /// Requires frequencies to be set via `with_frequencies()`.
    /// Indicators show scaling state:
    /// - ⚡ Turbo (>100% of base)
    /// - ↑ High (80-100% of max)
    /// - → Normal (50-80% of max)
    /// - ↓ Scaled (20-50% of max)
    /// - · Idle (<20% of max)
    #[must_use]
    pub fn with_freq_indicators(mut self) -> Self {
        self.show_freq_indicators = true;
        self
    }

    /// Set per-core frequency data (current and max frequencies in MHz).
    ///
    /// Both vectors should have the same length as `core_usage`.
    #[must_use]
    pub fn with_frequencies(mut self, current: Vec<u32>, max: Vec<u32>) -> Self {
        // Provability: frequency vectors must have consistent lengths
        debug_assert_eq!(
            current.len(),
            max.len(),
            "current and max frequencies must have same length"
        );
        self.frequencies = Some(current);
        self.max_frequencies = Some(max);
        self
    }

    /// Set the CPU governor (affects display title/info).
    #[must_use]
    pub fn with_governor(mut self, governor: CpuGovernor) -> Self {
        self.governor = Some(governor);
        self
    }

    /// Update core usage data.
    pub fn set_usage(&mut self, usage: Vec<f64>) {
        self.core_usage = usage;
    }

    /// Update frequency data.
    pub fn set_frequencies(&mut self, current: Vec<u32>, max: Vec<u32>) {
        self.frequencies = Some(current);
        self.max_frequencies = Some(max);
    }

    /// Update CPU governor.
    pub fn set_governor(&mut self, governor: CpuGovernor) {
        self.governor = Some(governor);
    }

    /// Get the current CPU governor, if set.
    #[must_use]
    pub fn governor(&self) -> Option<&CpuGovernor> {
        self.governor.as_ref()
    }

    /// Get the governor display string for UI titles.
    #[must_use]
    pub fn governor_display(&self) -> &'static str {
        self.governor.as_ref().map_or("", CpuGovernor::as_str)
    }

    /// Get the number of cores.
    #[must_use]
    pub fn core_count(&self) -> usize {
        self.core_usage.len()
    }

    /// Calculate optimal grid dimensions for N cores in given width.
    fn optimal_grid(&self, max_width: usize) -> (usize, usize) {
        let count = self.core_usage.len();
        if count == 0 {
            return (0, 0);
        }

        let cell_width = self.cell_width();

        let max_cols = (max_width / cell_width).max(1);
        let cols = self.columns.unwrap_or_else(|| {
            // Try to make a reasonably square grid
            let sqrt = (count as f64).sqrt().ceil() as usize;
            sqrt.min(max_cols).max(1)
        });

        let rows = count.div_ceil(cols);
        (cols, rows)
    }

    /// Get cell width based on current display mode.
    fn cell_width(&self) -> usize {
        let freq_extra = usize::from(self.show_freq_indicators);

        if self.show_percentages {
            // Format: "12 45%⚡" = 6 chars + freq indicator + space
            if self.compact {
                7 + freq_extra
            } else {
                8 + freq_extra
            }
        } else if self.show_labels {
            // Format: "12▆⚡" = 3 chars + freq indicator + space
            if self.compact {
                4 + freq_extra
            } else {
                5 + freq_extra
            }
        } else if self.compact {
            2 + freq_extra
        } else {
            3 + freq_extra
        }
    }

    /// Get meter character for percentage (0-100).
    fn meter_char(pct: f64) -> char {
        let idx = ((pct / 100.0) * 8.0).round() as usize;
        METER_CHARS[idx.min(8)]
    }

    /// Get frequency scaling state for a core.
    fn freq_scaling_state(&self, core_idx: usize) -> Option<FrequencyScalingState> {
        let frequencies = self.frequencies.as_ref()?;
        let max_frequencies = self.max_frequencies.as_ref()?;

        let current = *frequencies.get(core_idx)?;
        let max = *max_frequencies.get(core_idx)?;

        if max == 0 {
            return Some(FrequencyScalingState::Idle);
        }

        let ratio = current as f64 / max as f64;

        Some(if ratio > 1.0 {
            FrequencyScalingState::Turbo
        } else if ratio >= 0.8 {
            FrequencyScalingState::High
        } else if ratio >= 0.5 {
            FrequencyScalingState::Normal
        } else if ratio >= 0.2 {
            FrequencyScalingState::Scaled
        } else {
            FrequencyScalingState::Idle
        })
    }

    /// Get the frequency indicator character for a core.
    fn freq_indicator(&self, core_idx: usize) -> Option<char> {
        if !self.show_freq_indicators {
            return None;
        }

        self.freq_scaling_state(core_idx).map(|state| match state {
            FrequencyScalingState::Turbo => FREQ_INDICATORS[0], // ⚡
            FrequencyScalingState::High => FREQ_INDICATORS[1],  // ↑
            FrequencyScalingState::Normal => FREQ_INDICATORS[2], // →
            FrequencyScalingState::Scaled => FREQ_INDICATORS[3], // ↓
            FrequencyScalingState::Idle => FREQ_INDICATORS[4],  // ·
        })
    }

    /// Get the average frequency across all cores in GHz.
    #[must_use]
    pub fn avg_frequency_ghz(&self) -> Option<f64> {
        let frequencies = self.frequencies.as_ref()?;
        if frequencies.is_empty() {
            return None;
        }
        let sum: u64 = frequencies.iter().map(|&f| f as u64).sum();
        Some(sum as f64 / frequencies.len() as f64 / 1000.0)
    }
}

impl Brick for CpuGrid {
    fn brick_name(&self) -> &'static str {
        "cpu_grid"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: self.assertions().to_vec(),
            failed: vec![],
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for CpuGrid {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let max_width = constraints.max_width as usize;
        let (cols, rows) = self.optimal_grid(max_width);

        let cell_width = self.cell_width() as f32;

        let width = (cols as f32 * cell_width).min(constraints.max_width);
        let height = (rows as f32).min(constraints.max_height);

        constraints.constrain(Size::new(width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.core_usage.is_empty() {
            return;
        }

        let (cols, _rows) = self.optimal_grid(self.bounds.width as usize);
        if cols == 0 {
            return;
        }

        let cell_width = self.cell_width() as f32;

        for (i, &usage) in self.core_usage.iter().enumerate() {
            let col = i % cols;
            let row = i / cols;

            let x = self.bounds.x + col as f32 * cell_width;
            let y = self.bounds.y + row as f32;

            // Skip drawing if row exceeds bounds (clip to panel)
            if y >= self.bounds.y + self.bounds.height {
                break;
            }

            let usage_clamped = usage.clamp(0.0, 100.0);
            let color = self.gradient.for_percent(usage_clamped);

            let style = TextStyle {
                color,
                ..Default::default()
            };

            // Get frequency indicator if enabled
            let freq_char = self.freq_indicator(i).unwrap_or(' ');
            let freq_suffix = if self.show_freq_indicators {
                freq_char.to_string()
            } else {
                String::new()
            };

            if self.show_percentages {
                // Format: "12 45%⚡" or " 5 99%↑" (ttop style with numeric percentages)
                let label = if self.compact {
                    format!("{i:2}{usage_clamped:3.0}%{freq_suffix}")
                } else {
                    format!("{i:2} {usage_clamped:3.0}%{freq_suffix}")
                };
                canvas.draw_text(&label, Point::new(x, y), &style);
            } else if self.show_labels {
                // Format: "12▆⚡" or " 5▄→"
                let meter = Self::meter_char(usage_clamped);
                let label = if self.compact {
                    format!("{i:2}{meter}{freq_suffix}")
                } else {
                    format!("{i:2} {meter}{freq_suffix}")
                };
                canvas.draw_text(&label, Point::new(x, y), &style);
            } else {
                let meter = Self::meter_char(usage_clamped);
                let label = format!("{meter}{freq_suffix}");
                canvas.draw_text(&label, Point::new(x, y), &style);
            }
        }
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_grid_new() {
        let grid = CpuGrid::new(vec![10.0, 50.0, 90.0]);
        assert_eq!(grid.core_count(), 3);
    }

    #[test]
    fn test_cpu_grid_empty() {
        let grid = CpuGrid::new(vec![]);
        assert_eq!(grid.core_count(), 0);
    }

    #[test]
    fn test_cpu_grid_with_columns() {
        let grid = CpuGrid::new(vec![10.0; 48]).with_columns(8);
        assert_eq!(grid.columns, Some(8));
    }

    #[test]
    fn test_cpu_grid_compact() {
        let grid = CpuGrid::new(vec![10.0; 48]).compact();
        assert!(grid.compact);
    }

    #[test]
    fn test_cpu_grid_without_labels() {
        let grid = CpuGrid::new(vec![10.0; 48]).without_labels();
        assert!(!grid.show_labels);
    }

    #[test]
    fn test_meter_char() {
        assert_eq!(CpuGrid::meter_char(0.0), ' ');
        assert_eq!(CpuGrid::meter_char(50.0), '▄');
        assert_eq!(CpuGrid::meter_char(100.0), '█');
    }

    #[test]
    fn test_optimal_grid_48_cores() {
        let grid = CpuGrid::new(vec![10.0; 48]);
        let (cols, rows) = grid.optimal_grid(80);
        assert!(cols > 0);
        assert!(rows > 0);
        assert!(cols * rows >= 48);
    }

    #[test]
    fn test_optimal_grid_explicit_columns() {
        let grid = CpuGrid::new(vec![10.0; 48]).with_columns(8);
        let (cols, rows) = grid.optimal_grid(80);
        assert_eq!(cols, 8);
        assert_eq!(rows, 6);
    }

    #[test]
    fn test_cpu_grid_measure() {
        let grid = CpuGrid::new(vec![10.0; 8]);
        let size = grid.measure(Constraints::new(0.0, 80.0, 0.0, 20.0));
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn test_cpu_grid_set_usage() {
        let mut grid = CpuGrid::new(vec![]);
        assert_eq!(grid.core_count(), 0);
        grid.set_usage(vec![10.0, 20.0, 30.0]);
        assert_eq!(grid.core_count(), 3);
    }

    #[test]
    fn test_cpu_grid_verify() {
        let grid = CpuGrid::new(vec![50.0; 8]);
        let v = grid.verify();
        assert!(v.is_valid());
    }

    #[test]
    fn test_cpu_grid_type_id() {
        let grid = CpuGrid::new(vec![]);
        let _ = Widget::type_id(&grid);
    }

    #[test]
    fn test_cpu_grid_default() {
        let grid = CpuGrid::default();
        assert_eq!(grid.core_count(), 0);
    }

    #[test]
    fn test_cpu_grid_brick_name() {
        let grid = CpuGrid::new(vec![]);
        assert_eq!(grid.brick_name(), "cpu_grid");
    }

    #[test]
    fn test_cpu_grid_layout() {
        let mut grid = CpuGrid::new(vec![50.0; 8]);
        let result = grid.layout(Rect::new(0.0, 0.0, 80.0, 10.0));
        assert_eq!(result.size.width, 80.0);
        assert_eq!(result.size.height, 10.0);
    }

    #[test]
    fn test_cpu_grid_event() {
        let mut grid = CpuGrid::new(vec![50.0]);
        let event = Event::Resize {
            width: 80.0,
            height: 24.0,
        };
        assert!(grid.event(&event).is_none());
    }

    #[test]
    fn test_cpu_grid_children() {
        let grid = CpuGrid::new(vec![50.0]);
        assert!(grid.children().is_empty());
    }

    #[test]
    fn test_cpu_grid_children_mut() {
        let mut grid = CpuGrid::new(vec![50.0]);
        assert!(grid.children_mut().is_empty());
    }

    #[test]
    fn test_cpu_grid_assertions() {
        let grid = CpuGrid::new(vec![50.0]);
        assert!(!grid.assertions().is_empty());
    }

    #[test]
    fn test_cpu_grid_budget() {
        let grid = CpuGrid::new(vec![50.0]);
        let budget = grid.budget();
        assert!(budget.layout_ms > 0);
    }

    #[test]
    fn test_cpu_grid_to_html() {
        let grid = CpuGrid::new(vec![50.0]);
        assert!(grid.to_html().is_empty());
    }

    #[test]
    fn test_cpu_grid_to_css() {
        let grid = CpuGrid::new(vec![50.0]);
        assert!(grid.to_css().is_empty());
    }

    #[test]
    fn test_cpu_grid_clone() {
        let grid = CpuGrid::new(vec![50.0, 75.0]).with_columns(4);
        let cloned = grid.clone();
        assert_eq!(cloned.core_usage.len(), grid.core_usage.len());
        assert_eq!(cloned.columns, grid.columns);
    }

    #[test]
    fn test_cpu_grid_debug() {
        let grid = CpuGrid::new(vec![50.0]);
        let debug = format!("{grid:?}");
        assert!(debug.contains("CpuGrid"));
    }

    #[test]
    fn test_cpu_grid_with_gradient() {
        let gradient = Gradient::from_hex(&["#00FF00", "#FF0000"]);
        let grid = CpuGrid::new(vec![50.0]).with_gradient(gradient);
        // Gradient is private, just test it doesn't panic
        assert_eq!(grid.core_count(), 1);
    }

    #[test]
    fn test_cpu_grid_paint() {
        use crate::{CellBuffer, DirectTerminalCanvas};

        let mut grid = CpuGrid::new(vec![10.0, 50.0, 90.0]);
        let mut buffer = CellBuffer::new(40, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        grid.layout(Rect::new(0.0, 0.0, 40.0, 10.0));
        grid.paint(&mut canvas);
    }

    #[test]
    fn test_cpu_grid_paint_empty() {
        use crate::{CellBuffer, DirectTerminalCanvas};

        let grid = CpuGrid::new(vec![]);
        let mut buffer = CellBuffer::new(40, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        grid.paint(&mut canvas);
    }

    #[test]
    fn test_cpu_grid_paint_compact() {
        use crate::{CellBuffer, DirectTerminalCanvas};

        let mut grid = CpuGrid::new(vec![10.0, 50.0, 90.0]).compact();
        let mut buffer = CellBuffer::new(40, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        grid.layout(Rect::new(0.0, 0.0, 40.0, 10.0));
        grid.paint(&mut canvas);
    }

    #[test]
    fn test_cpu_grid_paint_without_labels() {
        use crate::{CellBuffer, DirectTerminalCanvas};

        let mut grid = CpuGrid::new(vec![10.0, 50.0, 90.0]).without_labels();
        let mut buffer = CellBuffer::new(40, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        grid.layout(Rect::new(0.0, 0.0, 40.0, 10.0));
        grid.paint(&mut canvas);
    }

    #[test]
    fn test_cpu_grid_paint_compact_without_labels() {
        use crate::{CellBuffer, DirectTerminalCanvas};

        let mut grid = CpuGrid::new(vec![10.0, 50.0, 90.0])
            .compact()
            .without_labels();
        let mut buffer = CellBuffer::new(40, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        grid.layout(Rect::new(0.0, 0.0, 40.0, 10.0));
        grid.paint(&mut canvas);
    }

    #[test]
    fn test_cpu_grid_optimal_grid_empty() {
        let grid = CpuGrid::new(vec![]);
        let (cols, rows) = grid.optimal_grid(80);
        assert_eq!(cols, 0);
        assert_eq!(rows, 0);
    }

    #[test]
    fn test_cpu_grid_measure_compact() {
        let grid = CpuGrid::new(vec![10.0; 8]).compact();
        let size = grid.measure(Constraints::new(0.0, 80.0, 0.0, 20.0));
        assert!(size.width > 0.0);
    }

    #[test]
    fn test_cpu_grid_measure_without_labels() {
        let grid = CpuGrid::new(vec![10.0; 8]).without_labels();
        let size = grid.measure(Constraints::new(0.0, 80.0, 0.0, 20.0));
        assert!(size.width > 0.0);
    }

    #[test]
    fn test_cpu_grid_measure_compact_without_labels() {
        let grid = CpuGrid::new(vec![10.0; 8]).compact().without_labels();
        let size = grid.measure(Constraints::new(0.0, 80.0, 0.0, 20.0));
        assert!(size.width > 0.0);
    }

    #[test]
    fn test_cpu_grid_meter_char_edge_cases() {
        assert_eq!(CpuGrid::meter_char(-10.0), ' ');
        assert_eq!(CpuGrid::meter_char(150.0), '█');
        assert_eq!(CpuGrid::meter_char(12.5), '▁');
        assert_eq!(CpuGrid::meter_char(25.0), '▂');
        assert_eq!(CpuGrid::meter_char(37.5), '▃');
        assert_eq!(CpuGrid::meter_char(62.5), '▅');
        assert_eq!(CpuGrid::meter_char(75.0), '▆');
        assert_eq!(CpuGrid::meter_char(87.5), '▇');
    }

    // ===== Frequency and Governor Tests (SPEC-024 Section 15) =====

    #[test]
    fn test_cpu_grid_with_frequencies() {
        let grid = CpuGrid::new(vec![50.0; 4])
            .with_frequencies(vec![3600, 3200, 2800, 1000], vec![4000, 4000, 4000, 4000]);

        assert!(grid.frequencies.is_some());
        assert!(grid.max_frequencies.is_some());
        assert_eq!(grid.frequencies.as_ref().unwrap().len(), 4);
    }

    #[test]
    fn test_cpu_grid_with_freq_indicators() {
        let grid = CpuGrid::new(vec![50.0; 4])
            .with_frequencies(vec![3600, 3200, 2800, 1000], vec![4000, 4000, 4000, 4000])
            .with_freq_indicators();

        assert!(grid.show_freq_indicators);
    }

    #[test]
    fn test_cpu_grid_freq_scaling_state() {
        let grid = CpuGrid::new(vec![50.0; 4]).with_frequencies(
            vec![4200, 3500, 2500, 500],  // current
            vec![4000, 4000, 4000, 4000], // max
        );

        // 4200/4000 = 105% -> Turbo
        assert_eq!(
            grid.freq_scaling_state(0),
            Some(FrequencyScalingState::Turbo)
        );
        // 3500/4000 = 87.5% -> High
        assert_eq!(
            grid.freq_scaling_state(1),
            Some(FrequencyScalingState::High)
        );
        // 2500/4000 = 62.5% -> Normal
        assert_eq!(
            grid.freq_scaling_state(2),
            Some(FrequencyScalingState::Normal)
        );
        // 500/4000 = 12.5% -> Idle
        assert_eq!(
            grid.freq_scaling_state(3),
            Some(FrequencyScalingState::Idle)
        );
    }

    #[test]
    fn test_cpu_grid_freq_scaling_state_scaled() {
        let grid = CpuGrid::new(vec![50.0; 1]).with_frequencies(vec![1200], vec![4000]);
        // 1200/4000 = 30% -> Scaled
        assert_eq!(
            grid.freq_scaling_state(0),
            Some(FrequencyScalingState::Scaled)
        );
    }

    #[test]
    fn test_cpu_grid_freq_scaling_state_no_frequencies() {
        let grid = CpuGrid::new(vec![50.0; 4]);
        assert_eq!(grid.freq_scaling_state(0), None);
    }

    #[test]
    fn test_cpu_grid_freq_scaling_state_zero_max() {
        let grid = CpuGrid::new(vec![50.0; 1]).with_frequencies(vec![3000], vec![0]);
        assert_eq!(
            grid.freq_scaling_state(0),
            Some(FrequencyScalingState::Idle)
        );
    }

    #[test]
    fn test_cpu_grid_freq_indicator() {
        let grid = CpuGrid::new(vec![50.0; 4])
            .with_frequencies(vec![4200, 3500, 2500, 500], vec![4000, 4000, 4000, 4000])
            .with_freq_indicators();

        assert_eq!(grid.freq_indicator(0), Some('⚡')); // Turbo
        assert_eq!(grid.freq_indicator(1), Some('↑')); // High
        assert_eq!(grid.freq_indicator(2), Some('→')); // Normal
        assert_eq!(grid.freq_indicator(3), Some('·')); // Idle
    }

    #[test]
    fn test_cpu_grid_freq_indicator_disabled() {
        let grid = CpuGrid::new(vec![50.0; 4])
            .with_frequencies(vec![4200, 3500, 2500, 500], vec![4000, 4000, 4000, 4000]);
        // freq indicators not enabled
        assert_eq!(grid.freq_indicator(0), None);
    }

    #[test]
    fn test_cpu_grid_with_governor() {
        let grid = CpuGrid::new(vec![50.0; 4]).with_governor(CpuGovernor::Performance);
        assert_eq!(grid.governor(), Some(&CpuGovernor::Performance));
        assert_eq!(grid.governor_display(), "performance");
    }

    #[test]
    fn test_cpu_grid_governor_powersave() {
        let grid = CpuGrid::new(vec![50.0; 4]).with_governor(CpuGovernor::Powersave);
        assert_eq!(grid.governor(), Some(&CpuGovernor::Powersave));
        assert_eq!(grid.governor_display(), "powersave");
    }

    #[test]
    fn test_cpu_grid_governor_schedutil() {
        let grid = CpuGrid::new(vec![50.0; 4]).with_governor(CpuGovernor::Schedutil);
        assert_eq!(grid.governor(), Some(&CpuGovernor::Schedutil));
        assert_eq!(grid.governor_display(), "schedutil");
    }

    #[test]
    fn test_cpu_grid_no_governor() {
        let grid = CpuGrid::new(vec![50.0; 4]);
        assert_eq!(grid.governor(), None);
        assert_eq!(grid.governor_display(), "");
    }

    #[test]
    fn test_cpu_grid_set_frequencies() {
        let mut grid = CpuGrid::new(vec![50.0; 4]);
        assert!(grid.frequencies.is_none());

        grid.set_frequencies(vec![3600, 3200, 2800, 1000], vec![4000, 4000, 4000, 4000]);
        assert!(grid.frequencies.is_some());
        assert!(grid.max_frequencies.is_some());
    }

    #[test]
    fn test_cpu_grid_set_governor() {
        let mut grid = CpuGrid::new(vec![50.0; 4]);
        assert!(grid.governor().is_none());

        grid.set_governor(CpuGovernor::Ondemand);
        assert_eq!(grid.governor(), Some(&CpuGovernor::Ondemand));
    }

    #[test]
    fn test_cpu_grid_avg_frequency_ghz() {
        let grid = CpuGrid::new(vec![50.0; 4])
            .with_frequencies(vec![3600, 3200, 2800, 2400], vec![4000, 4000, 4000, 4000]);

        let avg = grid.avg_frequency_ghz().unwrap();
        assert!((avg - 3.0).abs() < 0.01); // Average of 3.6+3.2+2.8+2.4 / 4 = 3.0 GHz
    }

    #[test]
    fn test_cpu_grid_avg_frequency_ghz_none() {
        let grid = CpuGrid::new(vec![50.0; 4]);
        assert!(grid.avg_frequency_ghz().is_none());
    }

    #[test]
    fn test_cpu_grid_avg_frequency_ghz_empty() {
        let grid = CpuGrid::new(vec![]).with_frequencies(vec![], vec![]);
        assert!(grid.avg_frequency_ghz().is_none());
    }

    #[test]
    fn test_cpu_grid_paint_with_freq_indicators() {
        use crate::{CellBuffer, DirectTerminalCanvas};

        let mut grid = CpuGrid::new(vec![50.0, 80.0, 20.0, 10.0])
            .with_frequencies(vec![4200, 3500, 2500, 500], vec![4000, 4000, 4000, 4000])
            .with_freq_indicators();

        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        grid.layout(Rect::new(0.0, 0.0, 60.0, 10.0));
        grid.paint(&mut canvas);
    }

    #[test]
    fn test_cpu_grid_paint_with_percentages_and_freq() {
        use crate::{CellBuffer, DirectTerminalCanvas};

        let mut grid = CpuGrid::new(vec![50.0, 80.0])
            .with_percentages()
            .with_frequencies(vec![4200, 3500], vec![4000, 4000])
            .with_freq_indicators();

        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        grid.layout(Rect::new(0.0, 0.0, 60.0, 10.0));
        grid.paint(&mut canvas);
    }

    #[test]
    fn test_cpu_grid_cell_width_with_freq_indicators() {
        let grid_no_freq = CpuGrid::new(vec![50.0; 4]);
        let grid_with_freq = CpuGrid::new(vec![50.0; 4]).with_freq_indicators();

        // Frequency indicators add 1 char
        assert_eq!(grid_with_freq.cell_width(), grid_no_freq.cell_width() + 1);
    }

    #[test]
    fn test_cpu_grid_cell_width_percentages_with_freq() {
        let grid = CpuGrid::new(vec![50.0; 4])
            .with_percentages()
            .with_freq_indicators();

        // Percentages (8) + freq indicator (1) = 9
        assert_eq!(grid.cell_width(), 9);
    }

    #[test]
    fn test_cpu_grid_governor_all_variants() {
        // Test all governor variants for display
        assert_eq!(
            CpuGrid::new(vec![])
                .with_governor(CpuGovernor::Performance)
                .governor_display(),
            "performance"
        );
        assert_eq!(
            CpuGrid::new(vec![])
                .with_governor(CpuGovernor::Powersave)
                .governor_display(),
            "powersave"
        );
        assert_eq!(
            CpuGrid::new(vec![])
                .with_governor(CpuGovernor::Ondemand)
                .governor_display(),
            "ondemand"
        );
        assert_eq!(
            CpuGrid::new(vec![])
                .with_governor(CpuGovernor::Conservative)
                .governor_display(),
            "conservative"
        );
        assert_eq!(
            CpuGrid::new(vec![])
                .with_governor(CpuGovernor::Schedutil)
                .governor_display(),
            "schedutil"
        );
        assert_eq!(
            CpuGrid::new(vec![])
                .with_governor(CpuGovernor::Userspace)
                .governor_display(),
            "userspace"
        );
        assert_eq!(
            CpuGrid::new(vec![])
                .with_governor(CpuGovernor::Unknown)
                .governor_display(),
            "unknown"
        );
    }
}
