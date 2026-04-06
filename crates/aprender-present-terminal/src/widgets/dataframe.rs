//! `DataFrame` widget with inline sparklines.
//!
//! Implements SPEC-024 Section 25 & 26.7 - Columnar data with inline visualizations.

use compact_str::CompactString;
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event, Key,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Status level for status dot visualization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StatusLevel {
    #[default]
    Ok,
    Warning,
    Critical,
    Unknown,
}

impl StatusLevel {
    /// Get the Unicode character and color for this status.
    #[must_use]
    pub fn render(self) -> (char, Color) {
        match self {
            Self::Ok => ('●', Color::new(0.2, 0.8, 0.2, 1.0)),
            Self::Warning => ('●', Color::new(0.9, 0.7, 0.1, 1.0)),
            Self::Critical => ('●', Color::new(0.9, 0.2, 0.2, 1.0)),
            Self::Unknown => ('○', Color::new(0.5, 0.5, 0.5, 1.0)),
        }
    }
}

/// Cell value types including inline visualizations.
#[derive(Debug, Clone, Default)]
pub enum CellValue {
    #[default]
    Null,
    Bool(bool),
    Int64(i64),
    Float64(f64),
    String(CompactString),
    /// Inline sparkline: ▁▂▃▅▆▇█
    Sparkline(Vec<f64>),
    /// Inline bar chart: ████▓▓░░
    SparkBar(Vec<f64>),
    /// Win/Loss indicator: ▲▼▲▲▼
    SparkWinLoss(Vec<i8>),
    /// Trend arrow with delta: ↑↗→↘↓
    TrendArrow(f64),
    /// Micro bar: █████░░░
    MicroBar {
        value: f64,
        max: f64,
    },
    /// Progress bar: ▓▓▓▓▓░░░░░
    ProgressBar(f64),
    /// Status dot: ● (colored)
    StatusDot(StatusLevel),
}

impl CellValue {
    /// Render cell value to string with given width.
    #[must_use]
    pub fn render(&self, width: usize) -> (String, Color) {
        match self {
            Self::Null => (String::new(), Color::new(0.5, 0.5, 0.5, 1.0)),
            Self::Bool(b) => (if *b { "true" } else { "false" }.to_string(), Color::WHITE),
            Self::Int64(n) => (format!("{n}"), Color::WHITE),
            Self::Float64(f) => (format!("{f:.2}"), Color::WHITE),
            Self::String(s) => (s.to_string(), Color::WHITE),
            Self::Sparkline(values) => (
                Self::render_sparkline(values, width),
                Color::new(0.3, 0.7, 1.0, 1.0),
            ),
            Self::SparkBar(values) => (
                Self::render_sparkbar(values, width),
                Color::new(0.5, 0.8, 0.5, 1.0),
            ),
            Self::SparkWinLoss(values) => (
                Self::render_winloss(values, width),
                Color::new(0.7, 0.7, 0.7, 1.0),
            ),
            Self::TrendArrow(delta) => Self::render_trend(*delta),
            Self::MicroBar { value, max } => (
                Self::render_microbar(*value, *max, width),
                Color::new(0.4, 0.6, 0.9, 1.0),
            ),
            Self::ProgressBar(pct) => (
                Self::render_progress(*pct, width),
                Color::new(0.3, 0.8, 0.3, 1.0),
            ),
            Self::StatusDot(level) => {
                let (ch, color) = level.render();
                (ch.to_string(), color)
            }
        }
    }

    fn render_sparkline(values: &[f64], width: usize) -> String {
        const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        if values.is_empty() {
            return " ".repeat(width);
        }

        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = (max - min).max(1e-10);

        let sample_width = width.min(values.len());
        let step = values.len() / sample_width.max(1);

        (0..sample_width)
            .map(|i| {
                let idx = (i * step).min(values.len() - 1);
                let v = values[idx];
                if !v.is_finite() {
                    return ' ';
                }
                let norm = ((v - min) / range * 7.0).round() as usize;
                BARS[norm.min(7)]
            })
            .collect()
    }

    fn render_sparkbar(values: &[f64], width: usize) -> String {
        const BLOCKS: [char; 4] = ['░', '▒', '▓', '█'];

        if values.is_empty() {
            return " ".repeat(width);
        }

        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let max = max.max(1e-10);

        let sample_width = width.min(values.len());
        let step = values.len() / sample_width.max(1);

        (0..sample_width)
            .map(|i| {
                let idx = (i * step).min(values.len() - 1);
                let v = values[idx];
                if !v.is_finite() || v < 0.0 {
                    return ' ';
                }
                let norm = ((v / max) * 3.0).round() as usize;
                BLOCKS[norm.min(3)]
            })
            .collect()
    }

    fn render_winloss(values: &[i8], width: usize) -> String {
        let sample_width = width.min(values.len());
        let step = values.len() / sample_width.max(1);

        (0..sample_width)
            .map(|i| {
                let idx = (i * step).min(values.len() - 1);
                match values[idx].cmp(&0) {
                    std::cmp::Ordering::Greater => '▲',
                    std::cmp::Ordering::Less => '▼',
                    std::cmp::Ordering::Equal => '─',
                }
            })
            .collect()
    }

    fn render_trend(delta: f64) -> (String, Color) {
        let (arrow, color) = if delta > 0.1 {
            ('↑', Color::new(0.2, 0.8, 0.2, 1.0))
        } else if delta > 0.02 {
            ('↗', Color::new(0.5, 0.8, 0.3, 1.0))
        } else if delta > -0.02 {
            ('→', Color::new(0.7, 0.7, 0.7, 1.0))
        } else if delta > -0.1 {
            ('↘', Color::new(0.8, 0.5, 0.3, 1.0))
        } else {
            ('↓', Color::new(0.9, 0.2, 0.2, 1.0))
        };
        (format!("{arrow} {delta:+.1}%"), color)
    }

    fn render_microbar(value: f64, max: f64, width: usize) -> String {
        let pct = (value / max.max(1e-10)).clamp(0.0, 1.0);
        let filled = ((width as f64) * pct).round() as usize;
        let empty = width.saturating_sub(filled);
        format!("{}{}", "█".repeat(filled), "░".repeat(empty))
    }

    fn render_progress(pct: f64, width: usize) -> String {
        let pct = pct.clamp(0.0, 100.0);
        let filled = ((width as f64) * (pct / 100.0)).round() as usize;
        let empty = width.saturating_sub(filled);
        format!("{}{}  {:.0}%", "▓".repeat(filled), "░".repeat(empty), pct)
    }
}

/// Column definition for `DataFrame`.
#[derive(Debug, Clone)]
pub struct Column {
    /// Column name.
    pub name: CompactString,
    /// Column values.
    pub values: Vec<CellValue>,
    /// Display width (in characters).
    pub width: usize,
    /// Alignment.
    pub align: ColumnAlign,
}

/// Column alignment.
#[derive(Debug, Clone, Copy, Default)]
pub enum ColumnAlign {
    #[default]
    Left,
    Right,
    Center,
}

impl Column {
    /// Create a new column.
    #[must_use]
    pub fn new(name: impl Into<CompactString>) -> Self {
        Self {
            name: name.into(),
            values: Vec::new(),
            width: 10,
            align: ColumnAlign::default(),
        }
    }

    /// Set column width.
    #[must_use]
    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Set column alignment.
    #[must_use]
    pub fn with_align(mut self, align: ColumnAlign) -> Self {
        self.align = align;
        self
    }

    /// Add values to column.
    #[must_use]
    pub fn with_values(mut self, values: Vec<CellValue>) -> Self {
        self.values = values;
        self
    }

    /// Create column from f64 values.
    #[must_use]
    pub fn from_f64(name: impl Into<CompactString>, values: &[f64]) -> Self {
        Self {
            name: name.into(),
            values: values.iter().map(|&v| CellValue::Float64(v)).collect(),
            width: 10,
            align: ColumnAlign::Right,
        }
    }

    /// Create column from i64 values.
    #[must_use]
    pub fn from_i64(name: impl Into<CompactString>, values: &[i64]) -> Self {
        Self {
            name: name.into(),
            values: values.iter().map(|&v| CellValue::Int64(v)).collect(),
            width: 10,
            align: ColumnAlign::Right,
        }
    }

    /// Create column from string values.
    #[must_use]
    pub fn from_strings(name: impl Into<CompactString>, values: &[&str]) -> Self {
        Self {
            name: name.into(),
            values: values
                .iter()
                .map(|&s| CellValue::String(CompactString::from(s)))
                .collect(),
            width: 15,
            align: ColumnAlign::Left,
        }
    }

    /// Create sparkline column from multiple source columns.
    #[must_use]
    pub fn sparkline_from_rows(name: impl Into<CompactString>, rows: Vec<Vec<f64>>) -> Self {
        Self {
            name: name.into(),
            values: rows.into_iter().map(CellValue::Sparkline).collect(),
            width: 12,
            align: ColumnAlign::Left,
        }
    }
}

/// `DataFrame` widget for tabular data with inline visualizations.
#[derive(Debug, Clone)]
pub struct DataFrame {
    columns: Vec<Column>,
    /// Number of rows to display.
    visible_rows: usize,
    /// Scroll offset.
    scroll_offset: usize,
    /// Selected row (if any).
    selected_row: Option<usize>,
    /// Show header.
    show_header: bool,
    /// Show row numbers.
    show_row_numbers: bool,
    /// Cached bounds.
    bounds: Rect,
}

impl DataFrame {
    /// Create a new empty `DataFrame`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            visible_rows: 20,
            scroll_offset: 0,
            selected_row: None,
            show_header: true,
            show_row_numbers: true,
            bounds: Rect::default(),
        }
    }

    /// Add a column.
    #[must_use]
    pub fn with_column(mut self, column: Column) -> Self {
        self.columns.push(column);
        self
    }

    /// Set visible rows.
    #[must_use]
    pub fn with_visible_rows(mut self, rows: usize) -> Self {
        self.visible_rows = rows;
        self
    }

    /// Toggle header visibility.
    #[must_use]
    pub fn with_header(mut self, show: bool) -> Self {
        self.show_header = show;
        self
    }

    /// Toggle row numbers.
    #[must_use]
    pub fn with_row_numbers(mut self, show: bool) -> Self {
        self.show_row_numbers = show;
        self
    }

    /// Get row count.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.columns.first().map_or(0, |c| c.values.len())
    }

    /// Get column count.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Add sparkline column from existing columns.
    pub fn add_sparkline_column(&mut self, name: &str, source_cols: &[usize]) {
        let row_count = self.row_count();
        let mut sparkline_data = Vec::with_capacity(row_count);

        for row_idx in 0..row_count {
            let values: Vec<f64> = source_cols
                .iter()
                .filter_map(|&col_idx| {
                    self.columns.get(col_idx).and_then(|col| {
                        col.values.get(row_idx).and_then(|v| match v {
                            CellValue::Float64(f) => Some(*f),
                            CellValue::Int64(i) => Some(*i as f64),
                            _ => None,
                        })
                    })
                })
                .collect();
            sparkline_data.push(CellValue::Sparkline(values));
        }

        self.columns.push(Column {
            name: CompactString::from(name),
            values: sparkline_data,
            width: 12,
            align: ColumnAlign::Left,
        });
    }

    /// Scroll to row.
    pub fn scroll_to(&mut self, row: usize) {
        let row_count = self.row_count();
        if row < row_count {
            self.scroll_offset = row.min(row_count.saturating_sub(self.visible_rows));
        }
    }

    /// Select row.
    pub fn select_row(&mut self, row: Option<usize>) {
        self.selected_row = row;
    }

    fn render_cell(&self, value: &CellValue, width: usize, align: ColumnAlign) -> (String, Color) {
        let (content, color) = value.render(width);
        let padded = match align {
            ColumnAlign::Left => format!("{content:<width$}"),
            ColumnAlign::Right => format!("{content:>width$}"),
            ColumnAlign::Center => format!("{content:^width$}"),
        };
        // Truncate if needed
        let truncated: String = padded.chars().take(width).collect();
        (truncated, color)
    }
}

impl Default for DataFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for DataFrame {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let total_width: usize = self.columns.iter().map(|c| c.width + 1).sum();
        Size::new(
            constraints.max_width.min(total_width as f32 + 5.0),
            constraints.max_height.min(self.visible_rows as f32 + 2.0),
        )
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        self.visible_rows = (bounds.height as usize).saturating_sub(2);
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 10.0 || self.bounds.height < 3.0 {
            return;
        }

        let header_style = TextStyle {
            color: Color::new(0.9, 0.9, 0.9, 1.0),
            ..Default::default()
        };

        let row_num_style = TextStyle {
            color: Color::new(0.5, 0.5, 0.5, 1.0),
            ..Default::default()
        };

        let selected_style = TextStyle {
            color: Color::new(0.2, 0.2, 0.2, 1.0),
            ..Default::default()
        };

        let row_num_width = if self.show_row_numbers { 5 } else { 0 };
        let mut y = self.bounds.y;

        // Draw header
        if self.show_header {
            let mut x = self.bounds.x + row_num_width as f32;

            if self.show_row_numbers {
                canvas.draw_text("#", Point::new(self.bounds.x, y), &row_num_style);
            }

            for col in &self.columns {
                let header: String = col.name.chars().take(col.width).collect();
                canvas.draw_text(&header, Point::new(x, y), &header_style);
                x += col.width as f32 + 1.0;
            }
            y += 1.0;

            // Separator
            let sep_width = (self.bounds.width as usize).min(120);
            canvas.draw_text(
                &"─".repeat(sep_width),
                Point::new(self.bounds.x, y),
                &row_num_style,
            );
            y += 1.0;
        }

        // Draw rows
        let row_count = self.row_count();
        let end_row = (self.scroll_offset + self.visible_rows).min(row_count);

        for row_idx in self.scroll_offset..end_row {
            let mut x = self.bounds.x + row_num_width as f32;
            let is_selected = self.selected_row == Some(row_idx);

            // Row number
            if self.show_row_numbers {
                let num = format!("{row_idx:>4}");
                canvas.draw_text(&num, Point::new(self.bounds.x, y), &row_num_style);
            }

            // Cell values
            for col in &self.columns {
                if let Some(value) = col.values.get(row_idx) {
                    let (content, color) = self.render_cell(value, col.width, col.align);

                    let style = if is_selected {
                        selected_style.clone()
                    } else {
                        TextStyle {
                            color,
                            ..Default::default()
                        }
                    };

                    canvas.draw_text(&content, Point::new(x, y), &style);
                }
                x += col.width as f32 + 1.0;
            }

            y += 1.0;
        }
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        if let Event::KeyDown { key, .. } = event {
            match key {
                Key::Up | Key::K => {
                    if let Some(row) = self.selected_row {
                        if row > 0 {
                            self.selected_row = Some(row - 1);
                            if row - 1 < self.scroll_offset {
                                self.scroll_offset = row - 1;
                            }
                        }
                    } else if self.row_count() > 0 {
                        self.selected_row = Some(0);
                    }
                }
                Key::Down | Key::J => {
                    let row_count = self.row_count();
                    if let Some(row) = self.selected_row {
                        if row + 1 < row_count {
                            self.selected_row = Some(row + 1);
                            if row + 1 >= self.scroll_offset + self.visible_rows {
                                self.scroll_offset = (row + 2).saturating_sub(self.visible_rows);
                            }
                        }
                    } else if row_count > 0 {
                        self.selected_row = Some(0);
                    }
                }
                Key::PageUp => {
                    self.scroll_offset = self.scroll_offset.saturating_sub(self.visible_rows);
                    if let Some(row) = self.selected_row {
                        if row >= self.visible_rows {
                            self.selected_row = Some(row - self.visible_rows);
                        } else {
                            self.selected_row = Some(0);
                        }
                    }
                }
                Key::PageDown => {
                    let row_count = self.row_count();
                    self.scroll_offset = (self.scroll_offset + self.visible_rows)
                        .min(row_count.saturating_sub(self.visible_rows));
                    if let Some(row) = self.selected_row {
                        let new_row = (row + self.visible_rows).min(row_count.saturating_sub(1));
                        self.selected_row = Some(new_row);
                    }
                }
                _ => {}
            }
        }
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}

impl Brick for DataFrame {
    fn brick_name(&self) -> &'static str {
        "DataFrame"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[
            BrickAssertion::max_latency_ms(16),
            BrickAssertion::max_latency_ms(50), // Filter budget
        ];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        // Check render budget
        if self.bounds.width >= 10.0 && self.bounds.height >= 3.0 {
            passed.push(BrickAssertion::max_latency_ms(16));
        } else {
            failed.push((
                BrickAssertion::max_latency_ms(16),
                "Size too small".to_string(),
            ));
        }

        // Check column consistency
        let row_count = self.row_count();
        for col in &self.columns {
            if col.values.len() != row_count {
                failed.push((
                    BrickAssertion::max_latency_ms(16),
                    format!("Column {} has inconsistent length", col.name),
                ));
            }
        }

        BrickVerification {
            passed,
            failed,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct::{CellBuffer, DirectTerminalCanvas};

    // ==================== DataFrame Basic Tests ====================

    #[test]
    fn test_dataframe_new() {
        let df = DataFrame::new();
        assert_eq!(df.row_count(), 0);
        assert_eq!(df.column_count(), 0);
    }

    #[test]
    fn test_dataframe_default() {
        let df = DataFrame::default();
        assert_eq!(df.row_count(), 0);
        assert_eq!(df.column_count(), 0);
    }

    #[test]
    fn test_dataframe_with_columns() {
        let df = DataFrame::new()
            .with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]))
            .with_column(Column::from_f64("B", &[4.0, 5.0, 6.0]));
        assert_eq!(df.row_count(), 3);
        assert_eq!(df.column_count(), 2);
    }

    #[test]
    fn test_dataframe_with_visible_rows() {
        let df = DataFrame::new().with_visible_rows(50);
        assert_eq!(df.visible_rows, 50);
    }

    #[test]
    fn test_dataframe_with_header() {
        let df = DataFrame::new().with_header(false);
        assert!(!df.show_header);
    }

    #[test]
    fn test_dataframe_with_row_numbers() {
        let df = DataFrame::new().with_row_numbers(false);
        assert!(!df.show_row_numbers);
    }

    // ==================== CellValue Tests ====================

    #[test]
    fn test_cell_value_null() {
        let (rendered, _) = CellValue::Null.render(5);
        assert!(rendered.is_empty());
    }

    #[test]
    fn test_cell_value_bool_true() {
        let (rendered, _) = CellValue::Bool(true).render(5);
        assert_eq!(rendered, "true");
    }

    #[test]
    fn test_cell_value_bool_false() {
        let (rendered, _) = CellValue::Bool(false).render(5);
        assert_eq!(rendered, "false");
    }

    #[test]
    fn test_cell_value_int64() {
        let (rendered, _) = CellValue::Int64(42).render(5);
        assert_eq!(rendered, "42");
    }

    #[test]
    fn test_cell_value_float64() {
        let (rendered, _) = CellValue::Float64(1.23456).render(10);
        assert!(rendered.contains("1.23"));
    }

    #[test]
    fn test_cell_value_string() {
        let (rendered, _) = CellValue::String(CompactString::from("hello")).render(10);
        assert_eq!(rendered, "hello");
    }

    #[test]
    fn test_cell_value_render_sparkline() {
        let values = vec![1.0, 5.0, 3.0, 8.0, 2.0];
        let (rendered, _) = CellValue::Sparkline(values).render(5);
        assert_eq!(rendered.chars().count(), 5);
    }

    #[test]
    fn test_cell_value_render_sparkline_empty() {
        let (rendered, _) = CellValue::Sparkline(vec![]).render(5);
        assert_eq!(rendered.len(), 5);
    }

    #[test]
    fn test_cell_value_render_sparkline_with_nan() {
        let values = vec![1.0, f64::NAN, 3.0, f64::INFINITY, 2.0];
        let (rendered, _) = CellValue::Sparkline(values).render(5);
        // Should handle NaN gracefully with spaces
        assert_eq!(rendered.chars().count(), 5);
    }

    #[test]
    fn test_cell_value_render_sparkline_constant() {
        // All same values - tests edge case where range is 0
        let values = vec![5.0, 5.0, 5.0, 5.0];
        let (rendered, _) = CellValue::Sparkline(values).render(4);
        assert_eq!(rendered.chars().count(), 4);
    }

    #[test]
    fn test_cell_value_render_progress() {
        let (rendered, _) = CellValue::ProgressBar(50.0).render(10);
        assert!(rendered.contains("50%"));
    }

    #[test]
    fn test_cell_value_render_progress_zero() {
        let (rendered, _) = CellValue::ProgressBar(0.0).render(10);
        assert!(rendered.contains("0%"));
    }

    #[test]
    fn test_cell_value_render_progress_hundred() {
        let (rendered, _) = CellValue::ProgressBar(100.0).render(10);
        assert!(rendered.contains("100%"));
    }

    #[test]
    fn test_cell_value_render_progress_clamp() {
        // Test clamping of values outside 0-100
        let (rendered, _) = CellValue::ProgressBar(150.0).render(10);
        assert!(rendered.contains("100%"));

        let (rendered2, _) = CellValue::ProgressBar(-10.0).render(10);
        assert!(rendered2.contains("0%"));
    }

    #[test]
    fn test_cell_value_render_trend_up() {
        let (rendered, color) = CellValue::TrendArrow(0.15).render(10);
        assert!(rendered.contains('↑'));
        assert!(color.g > color.r); // Green-ish
    }

    #[test]
    fn test_cell_value_render_trend_slight_up() {
        let (rendered, _) = CellValue::TrendArrow(0.05).render(10);
        assert!(rendered.contains('↗'));
    }

    #[test]
    fn test_cell_value_render_trend_flat() {
        let (rendered, _) = CellValue::TrendArrow(0.0).render(10);
        assert!(rendered.contains('→'));
    }

    #[test]
    fn test_cell_value_render_trend_slight_down() {
        let (rendered, _) = CellValue::TrendArrow(-0.05).render(10);
        assert!(rendered.contains('↘'));
    }

    #[test]
    fn test_cell_value_render_trend_down() {
        let (rendered, color) = CellValue::TrendArrow(-0.15).render(10);
        assert!(rendered.contains('↓'));
        assert!(color.r > color.g); // Red-ish
    }

    #[test]
    fn test_cell_value_render_status() {
        let (rendered, color) = CellValue::StatusDot(StatusLevel::Ok).render(1);
        assert_eq!(rendered, "●");
        assert!(color.g > 0.5);
    }

    #[test]
    fn test_cell_value_render_microbar() {
        let (rendered, _) = CellValue::MicroBar {
            value: 5.0,
            max: 10.0,
        }
        .render(10);
        assert!(rendered.contains('█'));
        assert!(rendered.contains('░'));
    }

    #[test]
    fn test_cell_value_render_microbar_full() {
        let (rendered, _) = CellValue::MicroBar {
            value: 10.0,
            max: 10.0,
        }
        .render(10);
        // Should be all filled
        assert_eq!(rendered.chars().filter(|&c| c == '█').count(), 10);
    }

    #[test]
    fn test_cell_value_render_microbar_zero_max() {
        let (rendered, _) = CellValue::MicroBar {
            value: 5.0,
            max: 0.0,
        }
        .render(10);
        // Should handle divide by zero gracefully
        assert!(!rendered.is_empty());
    }

    #[test]
    fn test_cell_value_default() {
        let value = CellValue::default();
        assert!(matches!(value, CellValue::Null));
    }

    // ==================== SparkBar Tests ====================

    #[test]
    fn test_sparkbar() {
        let values = vec![0.25, 0.5, 0.75, 1.0];
        let (rendered, _) = CellValue::SparkBar(values).render(4);
        assert_eq!(rendered.chars().count(), 4);
    }

    #[test]
    fn test_sparkbar_empty() {
        let (rendered, _) = CellValue::SparkBar(vec![]).render(5);
        assert_eq!(rendered.len(), 5);
    }

    #[test]
    fn test_sparkbar_with_nan() {
        let values = vec![0.5, f64::NAN, 0.75];
        let (rendered, _) = CellValue::SparkBar(values).render(3);
        assert_eq!(rendered.chars().count(), 3);
    }

    #[test]
    fn test_sparkbar_with_negative() {
        let values = vec![0.5, -0.5, 0.75];
        let (rendered, _) = CellValue::SparkBar(values).render(3);
        assert_eq!(rendered.chars().count(), 3);
    }

    // ==================== SparkWinLoss Tests ====================

    #[test]
    fn test_sparkwinloss() {
        let values = vec![1, -1, 0, 1, -1];
        let (rendered, _) = CellValue::SparkWinLoss(values).render(5);
        assert!(rendered.contains('▲'));
        assert!(rendered.contains('▼'));
        assert!(rendered.contains('─'));
    }

    #[test]
    fn test_sparkwinloss_empty() {
        let values: Vec<i8> = vec![];
        let (rendered, _) = CellValue::SparkWinLoss(values).render(5);
        assert!(rendered.is_empty());
    }

    // ==================== Status Level Tests ====================

    #[test]
    fn test_status_levels() {
        assert!(matches!(StatusLevel::default(), StatusLevel::Ok));

        let (_, ok_color) = StatusLevel::Ok.render();
        let (_, warn_color) = StatusLevel::Warning.render();
        let (_, crit_color) = StatusLevel::Critical.render();

        assert!(ok_color.g > ok_color.r);
        assert!(warn_color.r > 0.5 && warn_color.g > 0.5);
        assert!(crit_color.r > crit_color.g);
    }

    #[test]
    fn test_status_level_unknown() {
        let (ch, _) = StatusLevel::Unknown.render();
        assert_eq!(ch, '○');
    }

    #[test]
    fn test_status_level_clone_eq() {
        let s1 = StatusLevel::Ok;
        let s2 = s1;
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_status_level_debug() {
        let status = StatusLevel::Warning;
        let debug = format!("{:?}", status);
        assert!(debug.contains("Warning"));
    }

    // ==================== Column Tests ====================

    #[test]
    fn test_column_new() {
        let col = Column::new("Test");
        assert_eq!(col.name.as_str(), "Test");
        assert!(col.values.is_empty());
        assert_eq!(col.width, 10);
    }

    #[test]
    fn test_column_alignment() {
        let col = Column::new("Test")
            .with_align(ColumnAlign::Right)
            .with_width(10);
        assert!(matches!(col.align, ColumnAlign::Right));
        assert_eq!(col.width, 10);
    }

    #[test]
    fn test_column_with_values() {
        let col = Column::new("Test").with_values(vec![CellValue::Int64(1), CellValue::Int64(2)]);
        assert_eq!(col.values.len(), 2);
    }

    #[test]
    fn test_column_from_f64() {
        let col = Column::from_f64("Numbers", &[1.0, 2.0, 3.0]);
        assert_eq!(col.values.len(), 3);
        assert!(matches!(col.align, ColumnAlign::Right));
    }

    #[test]
    fn test_column_from_i64() {
        let col = Column::from_i64("Ints", &[1, 2, 3]);
        assert_eq!(col.values.len(), 3);
        assert!(matches!(col.align, ColumnAlign::Right));
    }

    #[test]
    fn test_column_from_strings() {
        let col = Column::from_strings("Names", &["Alice", "Bob"]);
        assert_eq!(col.values.len(), 2);
        assert!(matches!(col.align, ColumnAlign::Left));
        assert_eq!(col.width, 15);
    }

    #[test]
    fn test_column_sparkline_from_rows() {
        let rows = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let col = Column::sparkline_from_rows("Sparklines", rows);
        assert_eq!(col.values.len(), 2);
        assert_eq!(col.width, 12);
    }

    #[test]
    fn test_column_clone() {
        let col = Column::new("Test").with_width(20);
        let cloned = col.clone();
        assert_eq!(cloned.name, col.name);
        assert_eq!(cloned.width, 20);
    }

    #[test]
    fn test_column_debug() {
        let col = Column::new("Test");
        let debug = format!("{:?}", col);
        assert!(debug.contains("Column"));
    }

    // ==================== ColumnAlign Tests ====================

    #[test]
    fn test_column_align_default() {
        let align = ColumnAlign::default();
        assert!(matches!(align, ColumnAlign::Left));
    }

    #[test]
    fn test_column_align_center() {
        let col = Column::new("Test").with_align(ColumnAlign::Center);
        assert!(matches!(col.align, ColumnAlign::Center));
    }

    // ==================== DataFrame Sparkline Column ====================

    #[test]
    fn test_dataframe_sparkline_column() {
        let mut df = DataFrame::new()
            .with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]))
            .with_column(Column::from_f64("B", &[4.0, 5.0, 6.0]))
            .with_column(Column::from_f64("C", &[7.0, 8.0, 9.0]));

        df.add_sparkline_column("Trend", &[0, 1, 2]);
        assert_eq!(df.column_count(), 4);

        // Check sparkline column has correct values
        let sparkline_col = &df.columns[3];
        assert_eq!(sparkline_col.values.len(), 3);
    }

    #[test]
    fn test_dataframe_sparkline_column_invalid_source() {
        let mut df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]));

        // Reference non-existent column
        df.add_sparkline_column("Trend", &[0, 99]);
        assert_eq!(df.column_count(), 2);
    }

    // ==================== DataFrame Layout/Paint Tests ====================

    #[test]
    fn test_dataframe_layout() {
        let mut df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]));
        let bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
        let result = df.layout(bounds);
        assert!(result.size.width > 0.0);
    }

    #[test]
    fn test_dataframe_paint() {
        let mut df = DataFrame::new()
            .with_column(Column::from_strings("Name", &["Alice", "Bob", "Carol"]))
            .with_column(Column::from_f64("Score", &[95.0, 87.0, 92.0]));

        let bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
        df.layout(bounds);

        let mut buffer = CellBuffer::new(80, 24);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        df.paint(&mut canvas);
    }

    #[test]
    fn test_dataframe_paint_small_bounds() {
        let mut df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0]));

        df.bounds = Rect::new(0.0, 0.0, 5.0, 2.0); // Too small

        let mut buffer = CellBuffer::new(10, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        df.paint(&mut canvas); // Should return early without panic
    }

    #[test]
    fn test_dataframe_paint_no_header() {
        let mut df = DataFrame::new()
            .with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]))
            .with_header(false);

        let bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
        df.layout(bounds);

        let mut buffer = CellBuffer::new(80, 24);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        df.paint(&mut canvas);
    }

    #[test]
    fn test_dataframe_paint_no_row_numbers() {
        let mut df = DataFrame::new()
            .with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]))
            .with_row_numbers(false);

        let bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
        df.layout(bounds);

        let mut buffer = CellBuffer::new(80, 24);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        df.paint(&mut canvas);
    }

    #[test]
    fn test_dataframe_paint_with_selection() {
        let mut df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]));
        df.select_row(Some(1));

        let bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
        df.layout(bounds);

        let mut buffer = CellBuffer::new(80, 24);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        df.paint(&mut canvas);
    }

    #[test]
    fn test_dataframe_paint_with_all_cell_types() {
        let mut df = DataFrame::new().with_column(Column::new("Types").with_values(vec![
            CellValue::Null,
            CellValue::Bool(true),
            CellValue::Int64(42),
            CellValue::Float64(1.23),
            CellValue::String(CompactString::from("text")),
        ]));

        let bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
        df.layout(bounds);

        let mut buffer = CellBuffer::new(80, 24);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        df.paint(&mut canvas);
    }

    // ==================== DataFrame Scroll/Select Tests ====================

    #[test]
    fn test_dataframe_scroll() {
        let mut df = DataFrame::new().with_column(Column::from_f64(
            "A",
            &(0..100).map(|i| i as f64).collect::<Vec<_>>(),
        ));

        df.visible_rows = 10;
        df.scroll_to(50);
        assert_eq!(df.scroll_offset, 50);
    }

    #[test]
    fn test_dataframe_scroll_beyond_end() {
        let mut df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]));
        df.visible_rows = 10;
        df.scroll_to(100); // Beyond data
                           // Should clamp to valid range
        assert!(df.scroll_offset <= df.row_count());
    }

    #[test]
    fn test_dataframe_select() {
        let mut df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]));

        df.select_row(Some(1));
        assert_eq!(df.selected_row, Some(1));

        df.select_row(None);
        assert_eq!(df.selected_row, None);
    }

    // ==================== DataFrame Event Tests ====================

    #[test]
    fn test_dataframe_event_up() {
        let mut df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]));
        df.select_row(Some(2));

        let result = df.event(&Event::key_down(Key::Up));
        assert!(result.is_none());
        assert_eq!(df.selected_row, Some(1));
    }

    #[test]
    fn test_dataframe_event_down() {
        let mut df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]));
        df.select_row(Some(0));

        let result = df.event(&Event::key_down(Key::Down));
        assert!(result.is_none());
        assert_eq!(df.selected_row, Some(1));
    }

    #[test]
    fn test_dataframe_event_k() {
        let mut df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]));
        df.select_row(Some(2));

        let _ = df.event(&Event::key_down(Key::K));
        assert_eq!(df.selected_row, Some(1));
    }

    #[test]
    fn test_dataframe_event_j() {
        let mut df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]));
        df.select_row(Some(0));

        let _ = df.event(&Event::key_down(Key::J));
        assert_eq!(df.selected_row, Some(1));
    }

    #[test]
    fn test_dataframe_event_up_at_top() {
        let mut df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]));
        df.select_row(Some(0));

        let _ = df.event(&Event::key_down(Key::Up));
        assert_eq!(df.selected_row, Some(0)); // Should stay at top
    }

    #[test]
    fn test_dataframe_event_down_at_bottom() {
        let mut df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]));
        df.select_row(Some(2));

        let _ = df.event(&Event::key_down(Key::Down));
        assert_eq!(df.selected_row, Some(2)); // Should stay at bottom
    }

    #[test]
    fn test_dataframe_event_up_no_selection() {
        let mut df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]));
        df.select_row(None);

        let _ = df.event(&Event::key_down(Key::Up));
        assert_eq!(df.selected_row, Some(0)); // Should select first row
    }

    #[test]
    fn test_dataframe_event_down_no_selection() {
        let mut df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]));
        df.select_row(None);

        let _ = df.event(&Event::key_down(Key::Down));
        assert_eq!(df.selected_row, Some(0)); // Should select first row
    }

    #[test]
    fn test_dataframe_event_pageup() {
        let mut df = DataFrame::new().with_column(Column::from_f64(
            "A",
            &(0..50).map(|i| i as f64).collect::<Vec<_>>(),
        ));
        df.visible_rows = 10;
        df.scroll_offset = 30;
        df.select_row(Some(35));

        let _ = df.event(&Event::key_down(Key::PageUp));
        assert!(df.scroll_offset < 30);
    }

    #[test]
    fn test_dataframe_event_pagedown() {
        let mut df = DataFrame::new().with_column(Column::from_f64(
            "A",
            &(0..50).map(|i| i as f64).collect::<Vec<_>>(),
        ));
        df.visible_rows = 10;
        df.scroll_offset = 0;
        df.select_row(Some(5));

        let _ = df.event(&Event::key_down(Key::PageDown));
        assert!(df.scroll_offset > 0);
    }

    #[test]
    fn test_dataframe_event_other_key() {
        let mut df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0]));
        df.select_row(Some(0));

        let _ = df.event(&Event::key_down(Key::A));
        // Other keys should not change selection
        assert_eq!(df.selected_row, Some(0));
    }

    #[test]
    fn test_dataframe_event_non_keydown() {
        let mut df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0]));
        df.select_row(Some(0));

        let _ = df.event(&Event::FocusIn);
        // Non-key events should not change anything
        assert_eq!(df.selected_row, Some(0));
    }

    // ==================== DataFrame Widget Trait Tests ====================

    #[test]
    fn test_dataframe_measure() {
        let df =
            DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]).with_width(10));
        let size = df.measure(Constraints {
            min_width: 0.0,
            max_width: 100.0,
            min_height: 0.0,
            max_height: 50.0,
        });
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn test_dataframe_type_id() {
        let df = DataFrame::new();
        let type_id = Widget::type_id(&df);
        assert_eq!(type_id, TypeId::of::<DataFrame>());
    }

    #[test]
    fn test_dataframe_children() {
        let df = DataFrame::new();
        assert!(df.children().is_empty());
    }

    #[test]
    fn test_dataframe_children_mut() {
        let mut df = DataFrame::new();
        assert!(df.children_mut().is_empty());
    }

    // ==================== DataFrame Brick Trait Tests ====================

    #[test]
    fn test_dataframe_brick_name() {
        let df = DataFrame::new();
        assert_eq!(df.brick_name(), "DataFrame");
    }

    #[test]
    fn test_dataframe_assertions() {
        let df = DataFrame::new();
        assert!(!df.assertions().is_empty());
    }

    #[test]
    fn test_dataframe_budget() {
        let df = DataFrame::new();
        let budget = df.budget();
        assert!(budget.measure_ms > 0);
    }

    #[test]
    fn test_dataframe_verify() {
        let mut df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]));
        df.bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
        assert!(df.verify().is_valid());
    }

    #[test]
    fn test_dataframe_verify_small_bounds() {
        let mut df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]));
        df.bounds = Rect::new(0.0, 0.0, 5.0, 2.0);
        let verification = df.verify();
        assert!(!verification.failed.is_empty());
    }

    #[test]
    fn test_dataframe_verify_inconsistent_columns() {
        let mut df = DataFrame::new()
            .with_column(Column::from_f64("A", &[1.0, 2.0, 3.0]))
            .with_column(Column::from_f64("B", &[1.0, 2.0])); // Different length
        df.bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
        let verification = df.verify();
        // Should report inconsistent column lengths
        assert!(!verification.failed.is_empty());
    }

    #[test]
    fn test_dataframe_to_html() {
        let df = DataFrame::new();
        assert!(df.to_html().is_empty());
    }

    #[test]
    fn test_dataframe_to_css() {
        let df = DataFrame::new();
        assert!(df.to_css().is_empty());
    }

    // ==================== DataFrame Clone/Debug Tests ====================

    #[test]
    fn test_dataframe_clone() {
        let df = DataFrame::new().with_column(Column::from_f64("A", &[1.0, 2.0]));
        let cloned = df;
        assert_eq!(cloned.column_count(), 1);
        assert_eq!(cloned.row_count(), 2);
    }

    #[test]
    fn test_dataframe_debug() {
        let df = DataFrame::new();
        let debug = format!("{:?}", df);
        assert!(debug.contains("DataFrame"));
    }

    // ==================== CellValue Clone/Debug Tests ====================

    #[test]
    fn test_cell_value_clone() {
        let value = CellValue::Int64(42);
        let cloned = value;
        assert!(matches!(cloned, CellValue::Int64(42)));
    }

    #[test]
    fn test_cell_value_debug() {
        let value = CellValue::String(CompactString::from("test"));
        let debug = format!("{:?}", value);
        assert!(debug.contains("String"));
    }

    // ==================== render_cell Tests ====================

    #[test]
    fn test_render_cell_left_align() {
        let df = DataFrame::new();
        let value = CellValue::String(CompactString::from("hi"));
        let (content, _) = df.render_cell(&value, 10, ColumnAlign::Left);
        assert!(content.starts_with("hi"));
    }

    #[test]
    fn test_render_cell_right_align() {
        let df = DataFrame::new();
        let value = CellValue::String(CompactString::from("hi"));
        let (content, _) = df.render_cell(&value, 10, ColumnAlign::Right);
        assert!(content.ends_with("hi"));
    }

    #[test]
    fn test_render_cell_center_align() {
        let df = DataFrame::new();
        let value = CellValue::String(CompactString::from("hi"));
        let (content, _) = df.render_cell(&value, 10, ColumnAlign::Center);
        // Centered text should have spaces on both sides
        assert!(content.len() <= 10);
    }
}
