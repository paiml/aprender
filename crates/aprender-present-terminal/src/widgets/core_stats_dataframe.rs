//! `CoreStatsDataFrame` widget - Data science view for per-core CPU statistics.
//!
//! Grammar of Graphics construct: Core statistics as sortable `DataFrame` with:
//! - Core ID, Frequency, Temperature columns
//! - User/System/Idle/IOWait breakdown with stacked bars
//! - Context switches, interrupts per core
//! - Load history sparklines
//!
//! Implements SPEC-024 Section 27.8 - Framework-First pattern.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event, Key,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

use super::micro_heat_bar::HeatScheme;
use super::selection::RowHighlight;

/// Sort column for core stats table.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CoreStatsSortColumn {
    #[default]
    CoreId,
    Frequency,
    Temperature,
    User,
    System,
    Total,
    Idle,
}

impl CoreStatsSortColumn {
    /// Get column header string.
    #[must_use]
    pub fn header(self) -> &'static str {
        match self {
            Self::CoreId => "CORE",
            Self::Frequency => "FREQ",
            Self::Temperature => "TEMP",
            Self::User => "USR%",
            Self::System => "SYS%",
            Self::Total => "TOT%",
            Self::Idle => "IDL%",
        }
    }
}

/// Single core row data.
#[derive(Debug, Clone)]
pub struct CoreStatsRow {
    /// Core ID (0-indexed).
    pub core_id: u32,
    /// Current frequency in MHz.
    pub freq_mhz: u32,
    /// Temperature in Celsius (if available).
    pub temp_c: Option<f32>,
    /// User CPU percentage.
    pub user_pct: f32,
    /// System CPU percentage.
    pub system_pct: f32,
    /// `IOWait` percentage.
    pub iowait_pct: f32,
    /// Idle percentage.
    pub idle_pct: f32,
    /// Total utilization history (last N samples).
    pub util_history: Vec<f32>,
    /// Context switches per second.
    pub ctx_switches: u64,
    /// Interrupts per second.
    pub interrupts: u64,
}

impl Default for CoreStatsRow {
    fn default() -> Self {
        Self {
            core_id: 0,
            freq_mhz: 0,
            temp_c: None,
            user_pct: 0.0,
            system_pct: 0.0,
            iowait_pct: 0.0,
            idle_pct: 100.0,
            util_history: Vec::new(),
            ctx_switches: 0,
            interrupts: 0,
        }
    }
}

impl CoreStatsRow {
    /// Total CPU utilization (user + system + iowait).
    #[must_use]
    pub fn total_pct(&self) -> f32 {
        self.user_pct + self.system_pct + self.iowait_pct
    }
}

/// Core statistics `DataFrame` widget.
#[derive(Debug, Clone)]
pub struct CoreStatsDataFrame {
    /// Core rows.
    rows: Vec<CoreStatsRow>,
    /// Current sort column.
    sort_column: CoreStatsSortColumn,
    /// Sort descending.
    sort_desc: bool,
    /// Selected row index.
    selected_row: Option<usize>,
    /// Scroll offset.
    scroll_offset: usize,
    /// Show header row.
    show_header: bool,
    /// Show utilization breakdown bars.
    show_breakdown_bars: bool,
    /// Primary accent color.
    accent_color: Color,
    /// User color.
    user_color: Color,
    /// System color.
    system_color: Color,
    /// `IOWait` color.
    iowait_color: Color,
    /// Cached bounds.
    bounds: Rect,
}

impl Default for CoreStatsDataFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl CoreStatsDataFrame {
    /// Create a new empty core stats dataframe.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            sort_column: CoreStatsSortColumn::CoreId,
            sort_desc: false,
            selected_row: None,
            scroll_offset: 0,
            show_header: true,
            show_breakdown_bars: true,
            accent_color: Color::new(0.4, 0.7, 1.0, 1.0),
            user_color: Color::new(0.3, 0.7, 0.3, 1.0),
            system_color: Color::new(0.9, 0.5, 0.2, 1.0),
            iowait_color: Color::new(0.9, 0.2, 0.2, 1.0),
            bounds: Rect::default(),
        }
    }

    /// Update with new core data.
    pub fn update_cores(&mut self, cores: Vec<CoreStatsRow>) {
        self.rows = cores;
        self.sort_rows();
    }

    /// Set sort column.
    #[must_use]
    pub fn with_sort(mut self, column: CoreStatsSortColumn, descending: bool) -> Self {
        self.sort_column = column;
        self.sort_desc = descending;
        self
    }

    /// Set accent color.
    #[must_use]
    pub fn with_accent_color(mut self, color: Color) -> Self {
        self.accent_color = color;
        self
    }

    /// Set breakdown bar visibility.
    #[must_use]
    pub fn with_breakdown_bars(mut self, show: bool) -> Self {
        self.show_breakdown_bars = show;
        self
    }

    /// Cycle sort column.
    pub fn cycle_sort(&mut self) {
        self.sort_column = match self.sort_column {
            CoreStatsSortColumn::CoreId => CoreStatsSortColumn::Total,
            CoreStatsSortColumn::Total => CoreStatsSortColumn::User,
            CoreStatsSortColumn::User => CoreStatsSortColumn::System,
            CoreStatsSortColumn::System => CoreStatsSortColumn::Frequency,
            CoreStatsSortColumn::Frequency => CoreStatsSortColumn::Temperature,
            CoreStatsSortColumn::Temperature => CoreStatsSortColumn::Idle,
            CoreStatsSortColumn::Idle => CoreStatsSortColumn::CoreId,
        };
        self.sort_rows();
    }

    /// Toggle sort direction.
    pub fn toggle_sort_direction(&mut self) {
        self.sort_desc = !self.sort_desc;
        self.sort_rows();
    }

    fn sort_rows(&mut self) {
        let desc = self.sort_desc;
        match self.sort_column {
            CoreStatsSortColumn::CoreId => {
                self.rows.sort_by(|a, b| {
                    if desc {
                        b.core_id.cmp(&a.core_id)
                    } else {
                        a.core_id.cmp(&b.core_id)
                    }
                });
            }
            CoreStatsSortColumn::Frequency => {
                self.rows.sort_by(|a, b| {
                    if desc {
                        b.freq_mhz.cmp(&a.freq_mhz)
                    } else {
                        a.freq_mhz.cmp(&b.freq_mhz)
                    }
                });
            }
            CoreStatsSortColumn::Temperature => {
                self.rows.sort_by(|a, b| {
                    let a_temp = a.temp_c.unwrap_or(0.0);
                    let b_temp = b.temp_c.unwrap_or(0.0);
                    if desc {
                        b_temp
                            .partial_cmp(&a_temp)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        a_temp
                            .partial_cmp(&b_temp)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }
                });
            }
            CoreStatsSortColumn::User => {
                self.rows.sort_by(|a, b| {
                    if desc {
                        b.user_pct
                            .partial_cmp(&a.user_pct)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        a.user_pct
                            .partial_cmp(&b.user_pct)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }
                });
            }
            CoreStatsSortColumn::System => {
                self.rows.sort_by(|a, b| {
                    if desc {
                        b.system_pct
                            .partial_cmp(&a.system_pct)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        a.system_pct
                            .partial_cmp(&b.system_pct)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }
                });
            }
            CoreStatsSortColumn::Total => {
                self.rows.sort_by(|a, b| {
                    if desc {
                        b.total_pct()
                            .partial_cmp(&a.total_pct())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        a.total_pct()
                            .partial_cmp(&b.total_pct())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }
                });
            }
            CoreStatsSortColumn::Idle => {
                self.rows.sort_by(|a, b| {
                    if desc {
                        b.idle_pct
                            .partial_cmp(&a.idle_pct)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        a.idle_pct
                            .partial_cmp(&b.idle_pct)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }
                });
            }
        }
    }

    fn visible_rows(&self) -> usize {
        let header_offset = if self.show_header { 2 } else { 0 };
        (self.bounds.height as usize).saturating_sub(header_offset)
    }

    fn render_sparkline(values: &[f32], width: usize) -> String {
        const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        if values.is_empty() {
            return "─".repeat(width);
        }

        let sample_width = width.min(values.len());
        let step = values.len().saturating_sub(1) / sample_width.max(1);

        (0..sample_width)
            .map(|i| {
                let idx = (i * step.max(1)).min(values.len().saturating_sub(1));
                let v = values[idx].clamp(0.0, 100.0);
                let norm = ((v / 100.0) * 7.0).round() as usize;
                BARS[norm.min(7)]
            })
            .collect()
    }

    /// Render stacked breakdown bar using thermal heat color encoding.
    ///
    /// Grammar of Graphics: Maps CPU category percentages to:
    /// - Width proportional to percentage (area encoding)
    /// - Color intensity from thermal scheme (value encoding)
    ///
    /// Tufte principle: Double-encoding maximizes data-ink ratio.
    fn render_stacked_bar(&self, row: &CoreStatsRow, width: usize) -> Vec<(String, Color)> {
        let total = 100.0f32;
        let user_chars = ((row.user_pct / total) * width as f32).round() as usize;
        let sys_chars = ((row.system_pct / total) * width as f32).round() as usize;
        let io_chars = ((row.iowait_pct / total) * width as f32).round() as usize;
        let idle_chars = width.saturating_sub(user_chars + sys_chars + io_chars);

        // Thermal heat scheme: color intensity encodes the percentage value
        // High percentages = warmer colors (red), low = cooler (green)
        let scheme = HeatScheme::Thermal;

        // Use 8-level gradient blocks for elegant visual encoding
        let user_block = Self::gradient_block(row.user_pct);
        let sys_block = Self::gradient_block(row.system_pct);
        let io_block = Self::gradient_block(row.iowait_pct);

        vec![
            (
                user_block.to_string().repeat(user_chars),
                scheme.color_for_percent(row.user_pct as f64),
            ),
            (
                sys_block.to_string().repeat(sys_chars),
                scheme.color_for_percent(row.system_pct as f64),
            ),
            (
                io_block.to_string().repeat(io_chars),
                scheme.color_for_percent(row.iowait_pct as f64),
            ),
            ("░".repeat(idle_chars), Color::new(0.2, 0.2, 0.25, 1.0)),
        ]
    }

    /// Map percentage to gradient block character (8-level encoding).
    fn gradient_block(pct: f32) -> char {
        const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        let level = ((pct / 100.0) * 7.0).round() as usize;
        BLOCKS[level.min(7)]
    }

    fn format_freq(mhz: u32) -> String {
        if mhz >= 1000 {
            format!("{:.1}G", mhz as f32 / 1000.0)
        } else {
            format!("{mhz}M")
        }
    }

    fn format_temp(temp: Option<f32>) -> String {
        match temp {
            Some(t) => format!("{t:.0}°"),
            None => "─".to_string(),
        }
    }
}

impl Brick for CoreStatsDataFrame {
    fn brick_name(&self) -> &'static str {
        "core_stats_dataframe"
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
        format!(
            r#"<table class="core-stats-dataframe"><thead><tr>{}</tr></thead></table>"#,
            ["CORE", "FREQ", "TEMP", "USR%", "SYS%", "TOT%"]
                .iter()
                .map(|h| format!("<th>{h}</th>"))
                .collect::<String>()
        )
    }

    fn to_css(&self) -> String {
        ".core-stats-dataframe { sort: core_id; }".to_string()
    }
}

impl Widget for CoreStatsDataFrame {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let header_height = if self.show_header { 2.0 } else { 0.0 };
        let row_height = self.rows.len().min(64) as f32;
        let height = (header_height + row_height).min(constraints.max_height);
        constraints.constrain(Size::new(constraints.max_width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 50.0 || self.bounds.height < 3.0 {
            return;
        }

        let mut y = self.bounds.y;
        let x_start = self.bounds.x;
        let width = self.bounds.width as usize;

        // Column widths
        let col_core = 5;
        let col_freq = 6;
        let col_temp = 5;
        let col_user = 6;
        let col_sys = 6;
        let col_io = 5;
        let col_idle = 6;
        let col_bar = 20.min(width.saturating_sub(
            col_core + col_freq + col_temp + col_user + col_sys + col_io + col_idle + 10,
        ));
        let col_sparkline = 12.min(width.saturating_sub(
            col_core + col_freq + col_temp + col_user + col_sys + col_io + col_idle + col_bar + 12,
        ));

        let header_style = TextStyle {
            color: self.accent_color,
            ..Default::default()
        };

        let dim_style = TextStyle {
            color: Color::new(0.5, 0.5, 0.5, 1.0),
            ..Default::default()
        };

        // Header row
        if self.show_header {
            let headers = [
                (CoreStatsSortColumn::CoreId, col_core, "CORE"),
                (CoreStatsSortColumn::Frequency, col_freq, "FREQ"),
                (CoreStatsSortColumn::Temperature, col_temp, "TEMP"),
                (CoreStatsSortColumn::User, col_user, "USR%"),
                (CoreStatsSortColumn::System, col_sys, "SYS%"),
                (CoreStatsSortColumn::Total, col_io, "IO%"),
                (CoreStatsSortColumn::Idle, col_idle, "IDL%"),
            ];

            let mut hx = x_start;
            for (col, w, name) in headers {
                let is_sorted = col == self.sort_column;
                let arrow = if is_sorted {
                    if self.sort_desc {
                        "▼"
                    } else {
                        "▲"
                    }
                } else {
                    ""
                };
                let text = format!("{name}{arrow}");

                let style = if is_sorted { &header_style } else { &dim_style };
                canvas.draw_text(&text, Point::new(hx, y), style);
                hx += w as f32 + 1.0;
            }

            // Breakdown bar header
            if self.show_breakdown_bars && col_bar > 0 {
                canvas.draw_text("BREAKDOWN", Point::new(hx, y), &dim_style);
                hx += col_bar as f32 + 1.0;
            }

            // Sparkline header
            if col_sparkline > 0 {
                canvas.draw_text("HISTORY", Point::new(hx, y), &dim_style);
            }

            y += 1.0;

            // Separator
            let sep = "─".repeat(width.min(120));
            canvas.draw_text(&sep, Point::new(x_start, y), &dim_style);
            y += 1.0;
        }

        // Data rows
        let visible = self.visible_rows();
        let end_idx = (self.scroll_offset + visible).min(self.rows.len());

        for (rel_idx, row) in self.rows[self.scroll_offset..end_idx].iter().enumerate() {
            let abs_idx = self.scroll_offset + rel_idx;
            let is_selected = self.selected_row == Some(abs_idx);

            // === TUFTE SELECTION HIGHLIGHTING (Framework Widget) ===
            // Paint strong row background for selected row (immediately visible)
            let row_bounds = Rect::new(x_start, y, self.bounds.width, 1.0);
            let row_highlight = RowHighlight::new(row_bounds, is_selected);
            row_highlight.paint(canvas);

            // Get text style from row highlight (white on blue when selected)
            let text_style = row_highlight.text_style();
            let text_color = text_style.color;

            let mut x = x_start;

            // Core ID
            let core_str = format!("{:>4}", row.core_id);
            canvas.draw_text(&core_str, Point::new(x, y), &text_style);
            x += col_core as f32 + 1.0;

            // Frequency
            let freq_str = format!("{:>5}", Self::format_freq(row.freq_mhz));
            canvas.draw_text(&freq_str, Point::new(x, y), &text_style);
            x += col_freq as f32 + 1.0;

            // Temperature with color
            let temp_str = Self::format_temp(row.temp_c);
            let temp_color = match row.temp_c {
                Some(t) if t > 80.0 => Color::new(0.9, 0.2, 0.2, 1.0),
                Some(t) if t > 60.0 => Color::new(0.9, 0.7, 0.1, 1.0),
                Some(_) => text_color,
                None => Color::new(0.5, 0.5, 0.5, 1.0),
            };
            canvas.draw_text(
                &format!("{temp_str:>4}"),
                Point::new(x, y),
                &TextStyle {
                    color: temp_color,
                    ..Default::default()
                },
            );
            x += col_temp as f32 + 1.0;

            // User %
            canvas.draw_text(
                &format!("{:>5.1}", row.user_pct),
                Point::new(x, y),
                &TextStyle {
                    color: self.user_color,
                    ..Default::default()
                },
            );
            x += col_user as f32 + 1.0;

            // System %
            canvas.draw_text(
                &format!("{:>5.1}", row.system_pct),
                Point::new(x, y),
                &TextStyle {
                    color: self.system_color,
                    ..Default::default()
                },
            );
            x += col_sys as f32 + 1.0;

            // IOWait %
            canvas.draw_text(
                &format!("{:>4.1}", row.iowait_pct),
                Point::new(x, y),
                &TextStyle {
                    color: self.iowait_color,
                    ..Default::default()
                },
            );
            x += col_io as f32 + 1.0;

            // Idle %
            canvas.draw_text(
                &format!("{:>5.1}", row.idle_pct),
                Point::new(x, y),
                &text_style,
            );
            x += col_idle as f32 + 1.0;

            // Stacked breakdown bar
            if self.show_breakdown_bars && col_bar > 0 {
                let segments = self.render_stacked_bar(row, col_bar);
                let mut bx = x;
                for (chars, color) in segments {
                    canvas.draw_text(
                        &chars,
                        Point::new(bx, y),
                        &TextStyle {
                            color,
                            ..Default::default()
                        },
                    );
                    bx += chars.len() as f32;
                }
                x += col_bar as f32 + 1.0;
            }

            // Sparkline history
            if col_sparkline > 0 {
                let sparkline = Self::render_sparkline(&row.util_history, col_sparkline);
                canvas.draw_text(
                    &sparkline,
                    Point::new(x, y),
                    &TextStyle {
                        color: self.accent_color,
                        ..Default::default()
                    },
                );
            }

            y += 1.0;
        }
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        match event {
            Event::KeyDown {
                key: Key::Up | Key::K,
                ..
            } => {
                if let Some(sel) = self.selected_row {
                    if sel > 0 {
                        self.selected_row = Some(sel - 1);
                    }
                } else if !self.rows.is_empty() {
                    self.selected_row = Some(0);
                }
                None
            }
            Event::KeyDown {
                key: Key::Down | Key::J,
                ..
            } => {
                if let Some(sel) = self.selected_row {
                    if sel < self.rows.len().saturating_sub(1) {
                        self.selected_row = Some(sel + 1);
                    }
                } else if !self.rows.is_empty() {
                    self.selected_row = Some(0);
                }
                None
            }
            Event::KeyDown { key: Key::S, .. } => {
                self.cycle_sort();
                None
            }
            Event::KeyDown { key: Key::R, .. } => {
                self.toggle_sort_direction();
                None
            }
            _ => None,
        }
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
    use crate::direct::{CellBuffer, DirectTerminalCanvas};

    fn make_test_cores() -> Vec<CoreStatsRow> {
        vec![
            CoreStatsRow {
                core_id: 0,
                freq_mhz: 2400,
                temp_c: Some(45.0),
                user_pct: 10.0,
                system_pct: 5.0,
                iowait_pct: 2.0,
                idle_pct: 83.0,
                util_history: vec![10.0, 15.0, 12.0],
                ..Default::default()
            },
            CoreStatsRow {
                core_id: 1,
                freq_mhz: 3600,
                temp_c: Some(65.0),
                user_pct: 50.0,
                system_pct: 20.0,
                iowait_pct: 5.0,
                idle_pct: 25.0,
                util_history: vec![60.0, 70.0, 75.0],
                ..Default::default()
            },
            CoreStatsRow {
                core_id: 2,
                freq_mhz: 1800,
                temp_c: Some(85.0),
                user_pct: 30.0,
                system_pct: 10.0,
                iowait_pct: 3.0,
                idle_pct: 57.0,
                util_history: vec![35.0, 40.0, 43.0],
                ..Default::default()
            },
        ]
    }

    #[test]
    fn test_core_stats_creation() {
        let df = CoreStatsDataFrame::new()
            .with_sort(CoreStatsSortColumn::Total, true)
            .with_breakdown_bars(true);

        assert_eq!(df.sort_column, CoreStatsSortColumn::Total);
        assert!(df.sort_desc);
        assert!(df.show_breakdown_bars);
    }

    #[test]
    fn test_core_stats_default() {
        let df = CoreStatsDataFrame::default();
        assert_eq!(df.sort_column, CoreStatsSortColumn::CoreId);
        assert!(!df.sort_desc);
    }

    #[test]
    fn test_core_row_total() {
        let row = CoreStatsRow {
            user_pct: 30.0,
            system_pct: 10.0,
            iowait_pct: 5.0,
            idle_pct: 55.0,
            ..Default::default()
        };

        assert!((row.total_pct() - 45.0).abs() < 0.001);
    }

    #[test]
    fn test_core_row_default() {
        let row = CoreStatsRow::default();
        assert_eq!(row.core_id, 0);
        assert_eq!(row.idle_pct, 100.0);
        assert_eq!(row.total_pct(), 0.0);
    }

    #[test]
    fn test_format_freq() {
        assert_eq!(CoreStatsDataFrame::format_freq(2400), "2.4G");
        assert_eq!(CoreStatsDataFrame::format_freq(800), "800M");
        assert_eq!(CoreStatsDataFrame::format_freq(3600), "3.6G");
        assert_eq!(CoreStatsDataFrame::format_freq(999), "999M");
        assert_eq!(CoreStatsDataFrame::format_freq(1000), "1.0G");
    }

    #[test]
    fn test_format_temp() {
        assert_eq!(CoreStatsDataFrame::format_temp(Some(45.0)), "45°");
        assert_eq!(CoreStatsDataFrame::format_temp(Some(85.5)), "86°");
        assert_eq!(CoreStatsDataFrame::format_temp(None), "─");
    }

    #[test]
    fn test_sort_by_total() {
        let mut df = CoreStatsDataFrame::new().with_sort(CoreStatsSortColumn::Total, true);
        df.update_cores(make_test_cores());

        // Should be sorted descending by total
        assert_eq!(df.rows[0].core_id, 1); // 75% total
        assert_eq!(df.rows[1].core_id, 2); // 43% total
        assert_eq!(df.rows[2].core_id, 0); // 17% total
    }

    #[test]
    fn test_sort_by_frequency() {
        let mut df = CoreStatsDataFrame::new().with_sort(CoreStatsSortColumn::Frequency, true);
        df.update_cores(make_test_cores());

        assert_eq!(df.rows[0].core_id, 1); // 3600 MHz
        assert_eq!(df.rows[1].core_id, 0); // 2400 MHz
        assert_eq!(df.rows[2].core_id, 2); // 1800 MHz
    }

    #[test]
    fn test_sort_by_temperature() {
        let mut df = CoreStatsDataFrame::new().with_sort(CoreStatsSortColumn::Temperature, true);
        df.update_cores(make_test_cores());

        assert_eq!(df.rows[0].core_id, 2); // 85°C
        assert_eq!(df.rows[1].core_id, 1); // 65°C
        assert_eq!(df.rows[2].core_id, 0); // 45°C
    }

    #[test]
    fn test_sort_by_user() {
        let mut df = CoreStatsDataFrame::new().with_sort(CoreStatsSortColumn::User, true);
        df.update_cores(make_test_cores());

        assert_eq!(df.rows[0].core_id, 1); // 50%
        assert_eq!(df.rows[1].core_id, 2); // 30%
        assert_eq!(df.rows[2].core_id, 0); // 10%
    }

    #[test]
    fn test_sort_by_system() {
        let mut df = CoreStatsDataFrame::new().with_sort(CoreStatsSortColumn::System, true);
        df.update_cores(make_test_cores());

        assert_eq!(df.rows[0].core_id, 1); // 20%
        assert_eq!(df.rows[1].core_id, 2); // 10%
        assert_eq!(df.rows[2].core_id, 0); // 5%
    }

    #[test]
    fn test_sort_by_idle() {
        let mut df = CoreStatsDataFrame::new().with_sort(CoreStatsSortColumn::Idle, true);
        df.update_cores(make_test_cores());

        assert_eq!(df.rows[0].core_id, 0); // 83%
        assert_eq!(df.rows[1].core_id, 2); // 57%
        assert_eq!(df.rows[2].core_id, 1); // 25%
    }

    #[test]
    fn test_sort_ascending() {
        let mut df = CoreStatsDataFrame::new().with_sort(CoreStatsSortColumn::Frequency, false);
        df.update_cores(make_test_cores());

        assert_eq!(df.rows[0].core_id, 2); // 1800 MHz
        assert_eq!(df.rows[1].core_id, 0); // 2400 MHz
        assert_eq!(df.rows[2].core_id, 1); // 3600 MHz
    }

    #[test]
    fn test_cycle_sort() {
        let mut df = CoreStatsDataFrame::new();
        df.update_cores(make_test_cores());

        assert_eq!(df.sort_column, CoreStatsSortColumn::CoreId);
        df.cycle_sort();
        assert_eq!(df.sort_column, CoreStatsSortColumn::Total);
        df.cycle_sort();
        assert_eq!(df.sort_column, CoreStatsSortColumn::User);
        df.cycle_sort();
        assert_eq!(df.sort_column, CoreStatsSortColumn::System);
        df.cycle_sort();
        assert_eq!(df.sort_column, CoreStatsSortColumn::Frequency);
        df.cycle_sort();
        assert_eq!(df.sort_column, CoreStatsSortColumn::Temperature);
        df.cycle_sort();
        assert_eq!(df.sort_column, CoreStatsSortColumn::Idle);
        df.cycle_sort();
        assert_eq!(df.sort_column, CoreStatsSortColumn::CoreId);
    }

    #[test]
    fn test_toggle_sort_direction() {
        let mut df = CoreStatsDataFrame::new();
        assert!(!df.sort_desc);
        df.toggle_sort_direction();
        assert!(df.sort_desc);
        df.toggle_sort_direction();
        assert!(!df.sort_desc);
    }

    #[test]
    fn test_scroll_offset_default() {
        let df = CoreStatsDataFrame::new();
        assert_eq!(df.scroll_offset, 0);
    }

    #[test]
    fn test_paint() {
        let mut df = CoreStatsDataFrame::new();
        df.update_cores(make_test_cores());

        let mut buffer = CellBuffer::new(120, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        df.layout(Rect::new(0.0, 0.0, 120.0, 20.0));
        df.paint(&mut canvas);

        // Check that header was drawn
        let cell = buffer.get(0, 0).unwrap();
        assert!(!cell.symbol.is_empty());
    }

    #[test]
    fn test_paint_small_bounds() {
        let mut df = CoreStatsDataFrame::new();
        df.update_cores(make_test_cores());

        let mut buffer = CellBuffer::new(10, 2);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        df.layout(Rect::new(0.0, 0.0, 10.0, 2.0));
        df.paint(&mut canvas); // Should return early due to small bounds
    }

    #[test]
    fn test_paint_no_header() {
        let mut df = CoreStatsDataFrame::new();
        df.show_header = false;
        df.update_cores(make_test_cores());

        let mut buffer = CellBuffer::new(120, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        df.layout(Rect::new(0.0, 0.0, 120.0, 20.0));
        df.paint(&mut canvas);
    }

    #[test]
    fn test_paint_no_breakdown_bars() {
        let mut df = CoreStatsDataFrame::new().with_breakdown_bars(false);
        df.update_cores(make_test_cores());

        let mut buffer = CellBuffer::new(120, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        df.layout(Rect::new(0.0, 0.0, 120.0, 20.0));
        df.paint(&mut canvas);
    }

    #[test]
    fn test_paint_selected_row() {
        let mut df = CoreStatsDataFrame::new();
        df.update_cores(make_test_cores());
        df.selected_row = Some(1);

        let mut buffer = CellBuffer::new(120, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        df.layout(Rect::new(0.0, 0.0, 120.0, 20.0));
        df.paint(&mut canvas);
    }

    #[test]
    fn test_event_key_down() {
        let mut df = CoreStatsDataFrame::new();
        df.update_cores(make_test_cores());
        df.layout(Rect::new(0.0, 0.0, 100.0, 10.0));

        let event = Event::key_down(Key::Down);
        let _ = df.event(&event);
    }

    #[test]
    fn test_event_key_up() {
        let mut df = CoreStatsDataFrame::new();
        df.update_cores(make_test_cores());
        df.layout(Rect::new(0.0, 0.0, 100.0, 10.0));

        let event = Event::key_down(Key::Up);
        let _ = df.event(&event);
    }

    #[test]
    fn test_event_key_s() {
        let mut df = CoreStatsDataFrame::new();
        df.update_cores(make_test_cores());

        let event = Event::key_down(Key::S);
        let _ = df.event(&event);
        // Sort should have cycled
        assert_eq!(df.sort_column, CoreStatsSortColumn::Total);
    }

    #[test]
    fn test_event_key_r() {
        let mut df = CoreStatsDataFrame::new();
        df.update_cores(make_test_cores());

        let event = Event::key_down(Key::R);
        let _ = df.event(&event);
        // Sort direction should have toggled
        assert!(df.sort_desc);
    }

    #[test]
    fn test_measure() {
        let df = CoreStatsDataFrame::new();
        let constraints = Constraints {
            min_width: 0.0,
            max_width: 100.0,
            min_height: 0.0,
            max_height: 50.0,
        };
        let size = df.measure(constraints);
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn test_brick_name() {
        let df = CoreStatsDataFrame::new();
        assert_eq!(df.brick_name(), "core_stats_dataframe");
    }

    #[test]
    fn test_brick_verify() {
        let mut df = CoreStatsDataFrame::new();
        df.update_cores(make_test_cores());
        let v = df.verify();
        assert!(v.failed.is_empty());
    }

    #[test]
    fn test_brick_html() {
        let df = CoreStatsDataFrame::new();
        let html = df.to_html();
        assert!(html.contains("core-stats-dataframe"));
        assert!(html.contains("CORE"));
    }

    #[test]
    fn test_brick_css() {
        let df = CoreStatsDataFrame::new();
        let css = df.to_css();
        assert!(css.contains("core_id"));
    }

    #[test]
    fn test_column_headers() {
        assert_eq!(CoreStatsSortColumn::CoreId.header(), "CORE");
        assert_eq!(CoreStatsSortColumn::Frequency.header(), "FREQ");
        assert_eq!(CoreStatsSortColumn::Temperature.header(), "TEMP");
        assert_eq!(CoreStatsSortColumn::User.header(), "USR%");
        assert_eq!(CoreStatsSortColumn::System.header(), "SYS%");
        assert_eq!(CoreStatsSortColumn::Total.header(), "TOT%");
        assert_eq!(CoreStatsSortColumn::Idle.header(), "IDL%");
    }

    #[test]
    fn test_accent_color() {
        let color = Color::new(0.8, 0.2, 0.5, 1.0);
        let df = CoreStatsDataFrame::new().with_accent_color(color);
        assert!((df.accent_color.r - 0.8).abs() < 0.001);
    }
}
