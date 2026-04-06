//! DataFrame widget with inline visualizations.
//!
//! Provides a tabular data widget with support for inline sparklines,
//! progress bars, status indicators, and trend arrows.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::Widget,
};

/// Status level for status dot visualization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StatusLevel {
    /// OK status (green).
    #[default]
    Ok,
    /// Warning status (yellow).
    Warning,
    /// Critical status (red).
    Critical,
    /// Unknown status (gray).
    Unknown,
}

impl StatusLevel {
    /// Get the character and color for this status.
    #[must_use]
    pub fn render(self) -> (char, Color) {
        match self {
            Self::Ok => ('●', Color::Green),
            Self::Warning => ('●', Color::Yellow),
            Self::Critical => ('●', Color::Red),
            Self::Unknown => ('○', Color::DarkGray),
        }
    }
}

/// Cell value types including inline visualizations.
#[derive(Debug, Clone, Default)]
pub enum CellValue {
    /// Null/empty value.
    #[default]
    Null,
    /// Boolean value.
    Bool(bool),
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f64),
    /// String value.
    Text(String),
    /// Inline sparkline: ▁▂▃▄▅▆▇█
    Sparkline(Vec<f64>),
    /// Progress bar: ▓▓▓░░░ 50%
    Progress(f64),
    /// Status dot with color.
    Status(StatusLevel),
    /// Trend arrow with delta: ↑+5.2%
    Trend(f64),
    /// Micro bar: ████░░░░
    MicroBar {
        /// Current value.
        value: f64,
        /// Maximum value.
        max: f64,
    },
}

impl CellValue {
    /// Render cell value to string and color.
    #[must_use]
    pub fn render(&self, width: usize) -> (String, Color) {
        match self {
            Self::Null => (String::new(), Color::DarkGray),
            Self::Bool(b) => (if *b { "true" } else { "false" }.to_string(), Color::White),
            Self::Int(n) => (n.to_string(), Color::White),
            Self::Float(f) => (format!("{f:.2}"), Color::White),
            Self::Text(s) => (s.clone(), Color::White),
            Self::Sparkline(values) => (Self::render_sparkline(values, width), Color::Cyan),
            Self::Progress(pct) => (Self::render_progress(*pct, width), Color::Green),
            Self::Status(level) => {
                let (ch, color) = level.render();
                (ch.to_string(), color)
            }
            Self::Trend(delta) => Self::render_trend(*delta),
            Self::MicroBar { value, max } => {
                (Self::render_microbar(*value, *max, width), Color::Blue)
            }
        }
    }

    fn render_sparkline(values: &[f64], width: usize) -> String {
        const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        if values.is_empty() {
            return " ".repeat(width);
        }

        let min = values.iter().filter(|x| x.is_finite()).copied().fold(f64::INFINITY, f64::min);
        let max =
            values.iter().filter(|x| x.is_finite()).copied().fold(f64::NEG_INFINITY, f64::max);
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

    fn render_progress(pct: f64, width: usize) -> String {
        let pct = pct.clamp(0.0, 100.0);
        let bar_width = width.saturating_sub(5);
        let filled = ((bar_width as f64) * (pct / 100.0)).round() as usize;
        let empty = bar_width.saturating_sub(filled);
        format!("{}{}{:>3.0}%", "▓".repeat(filled), "░".repeat(empty), pct)
    }

    fn render_trend(delta: f64) -> (String, Color) {
        let (arrow, color) = if delta > 0.1 {
            ('↑', Color::Green)
        } else if delta > 0.02 {
            ('↗', Color::LightGreen)
        } else if delta > -0.02 {
            ('→', Color::DarkGray)
        } else if delta > -0.1 {
            ('↘', Color::LightRed)
        } else {
            ('↓', Color::Red)
        };
        (format!("{arrow}{delta:+.1}%"), color)
    }

    fn render_microbar(value: f64, max: f64, width: usize) -> String {
        let pct = (value / max.max(1e-10)).clamp(0.0, 1.0);
        let filled = ((width as f64) * pct).round() as usize;
        let empty = width.saturating_sub(filled);
        format!("{}{}", "█".repeat(filled), "░".repeat(empty))
    }
}

/// Column alignment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ColumnAlign {
    /// Left-aligned (default for text).
    #[default]
    Left,
    /// Right-aligned (default for numbers).
    Right,
    /// Center-aligned.
    Center,
}

/// Column definition for DataFrame.
#[derive(Debug, Clone)]
pub struct Column {
    /// Column name/header.
    pub name: String,
    /// Column values.
    pub values: Vec<CellValue>,
    /// Display width in characters.
    pub width: usize,
    /// Alignment.
    pub align: ColumnAlign,
}

impl Column {
    /// Create a new column.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), values: Vec::new(), width: 10, align: ColumnAlign::default() }
    }

    /// Set column width.
    #[must_use]
    pub fn width(mut self, width: usize) -> Self {
        self.width = width.max(3);
        self
    }

    /// Set alignment.
    #[must_use]
    pub fn align(mut self, align: ColumnAlign) -> Self {
        self.align = align;
        self
    }

    /// Set values.
    #[must_use]
    pub fn values(mut self, values: Vec<CellValue>) -> Self {
        self.values = values;
        self
    }

    /// Create column from f64 values.
    #[must_use]
    pub fn from_f64(name: impl Into<String>, values: &[f64]) -> Self {
        Self {
            name: name.into(),
            values: values.iter().map(|&v| CellValue::Float(v)).collect(),
            width: 10,
            align: ColumnAlign::Right,
        }
    }

    /// Create column from i64 values.
    #[must_use]
    pub fn from_i64(name: impl Into<String>, values: &[i64]) -> Self {
        Self {
            name: name.into(),
            values: values.iter().map(|&v| CellValue::Int(v)).collect(),
            width: 10,
            align: ColumnAlign::Right,
        }
    }

    /// Create column from strings.
    #[must_use]
    pub fn from_strings(name: impl Into<String>, values: &[&str]) -> Self {
        Self {
            name: name.into(),
            values: values.iter().map(|&s| CellValue::Text(s.to_string())).collect(),
            width: 15,
            align: ColumnAlign::Left,
        }
    }

    /// Create sparkline column from row data.
    #[must_use]
    pub fn sparklines(name: impl Into<String>, rows: Vec<Vec<f64>>) -> Self {
        Self {
            name: name.into(),
            values: rows.into_iter().map(CellValue::Sparkline).collect(),
            width: 12,
            align: ColumnAlign::Left,
        }
    }
}

/// DataFrame widget for tabular data with inline visualizations.
#[derive(Debug, Clone)]
pub struct DataFrame {
    /// Columns.
    columns: Vec<Column>,
    /// Number of visible rows.
    visible_rows: usize,
    /// Scroll offset.
    scroll_offset: usize,
    /// Selected row.
    selected_row: Option<usize>,
    /// Show header row.
    show_header: bool,
    /// Show row numbers.
    show_row_numbers: bool,
    /// Title.
    title: Option<String>,
}

impl Default for DataFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl DataFrame {
    /// Create a new empty DataFrame.
    #[must_use]
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            visible_rows: 20,
            scroll_offset: 0,
            selected_row: None,
            show_header: true,
            show_row_numbers: true,
            title: None,
        }
    }

    /// Add a column.
    #[must_use]
    pub fn column(mut self, column: Column) -> Self {
        self.columns.push(column);
        self
    }

    /// Set visible rows.
    #[must_use]
    pub fn visible_rows(mut self, rows: usize) -> Self {
        self.visible_rows = rows;
        self
    }

    /// Toggle header visibility.
    #[must_use]
    pub fn show_header(mut self, show: bool) -> Self {
        self.show_header = show;
        self
    }

    /// Toggle row numbers.
    #[must_use]
    pub fn show_row_numbers(mut self, show: bool) -> Self {
        self.show_row_numbers = show;
        self
    }

    /// Set title.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
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

    /// Select a row.
    pub fn select_row(&mut self, row: Option<usize>) {
        self.selected_row = row;
    }

    /// Scroll to a row.
    pub fn scroll_to(&mut self, row: usize) {
        let row_count = self.row_count();
        if row < row_count {
            self.scroll_offset = row.min(row_count.saturating_sub(self.visible_rows));
        }
    }

    fn render_cell(&self, value: &CellValue, width: usize, align: ColumnAlign) -> (String, Color) {
        let (content, color) = value.render(width);
        let padded = match align {
            ColumnAlign::Left => format!("{content:<width$}"),
            ColumnAlign::Right => format!("{content:>width$}"),
            ColumnAlign::Center => format!("{content:^width$}"),
        };
        let truncated: String = padded.chars().take(width).collect();
        (truncated, color)
    }
}

impl Widget for DataFrame {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width < 10 || area.height < 3 || self.columns.is_empty() {
            return;
        }

        let header_style = Style::default().fg(Color::White);
        let row_num_style = Style::default().fg(Color::DarkGray);
        let selected_style = Style::default().fg(Color::Black).bg(Color::White);

        let row_num_width: u16 = if self.show_row_numbers { 5 } else { 0 };
        let mut y = area.y;

        // Draw title
        if let Some(ref title) = self.title {
            for (i, ch) in title.chars().enumerate() {
                let x = area.x + i as u16;
                if x < area.x + area.width {
                    if let Some(cell) = buf.cell_mut((x, y)) {
                        cell.set_char(ch).set_style(header_style);
                    }
                }
            }
            y += 1;
        }

        // Draw header
        if self.show_header {
            let mut x = area.x + row_num_width;

            if self.show_row_numbers {
                let header = "#";
                for (i, ch) in header.chars().enumerate() {
                    let hx = area.x + i as u16;
                    if hx < area.x + area.width {
                        if let Some(cell) = buf.cell_mut((hx, y)) {
                            cell.set_char(ch).set_style(row_num_style);
                        }
                    }
                }
            }

            for col in &self.columns {
                let header: String = col.name.chars().take(col.width).collect();
                for (i, ch) in header.chars().enumerate() {
                    let hx = x + i as u16;
                    if hx < area.x + area.width {
                        if let Some(cell) = buf.cell_mut((hx, y)) {
                            cell.set_char(ch).set_style(header_style);
                        }
                    }
                }
                x += col.width as u16 + 1;
            }
            y += 1;

            // Separator
            for i in 0..area.width {
                let sx = area.x + i;
                if let Some(cell) = buf.cell_mut((sx, y)) {
                    cell.set_char('─').set_style(row_num_style);
                }
            }
            y += 1;
        }

        // Draw rows
        let row_count = self.row_count();
        let end_row = (self.scroll_offset + self.visible_rows).min(row_count);

        for row_idx in self.scroll_offset..end_row {
            if y >= area.y + area.height {
                break;
            }

            let mut x = area.x + row_num_width;
            let is_selected = self.selected_row == Some(row_idx);

            // Row number
            if self.show_row_numbers {
                let num = format!("{row_idx:>4}");
                for (i, ch) in num.chars().enumerate() {
                    let nx = area.x + i as u16;
                    if nx < area.x + area.width {
                        if let Some(cell) = buf.cell_mut((nx, y)) {
                            cell.set_char(ch).set_style(row_num_style);
                        }
                    }
                }
            }

            // Cell values
            for col in &self.columns {
                if let Some(value) = col.values.get(row_idx) {
                    let (content, color) = self.render_cell(value, col.width, col.align);

                    let style =
                        if is_selected { selected_style } else { Style::default().fg(color) };

                    for (i, ch) in content.chars().enumerate() {
                        let cx = x + i as u16;
                        if cx < area.x + area.width {
                            if let Some(cell) = buf.cell_mut((cx, y)) {
                                cell.set_char(ch).set_style(style);
                            }
                        }
                    }
                }
                x += col.width as u16 + 1;
            }

            y += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod status_level_tests {
        use super::*;

        #[test]
        fn test_default() {
            assert_eq!(StatusLevel::default(), StatusLevel::Ok);
        }

        #[test]
        fn test_render_ok() {
            let (ch, color) = StatusLevel::Ok.render();
            assert_eq!(ch, '●');
            assert_eq!(color, Color::Green);
        }

        #[test]
        fn test_render_warning() {
            let (ch, color) = StatusLevel::Warning.render();
            assert_eq!(ch, '●');
            assert_eq!(color, Color::Yellow);
        }

        #[test]
        fn test_render_critical() {
            let (ch, color) = StatusLevel::Critical.render();
            assert_eq!(ch, '●');
            assert_eq!(color, Color::Red);
        }

        #[test]
        fn test_render_unknown() {
            let (ch, color) = StatusLevel::Unknown.render();
            assert_eq!(ch, '○');
            assert_eq!(color, Color::DarkGray);
        }

        #[test]
        fn test_equality() {
            assert_eq!(StatusLevel::Ok, StatusLevel::Ok);
            assert_ne!(StatusLevel::Ok, StatusLevel::Warning);
        }
    }

    mod cell_value_tests {
        use super::*;

        #[test]
        fn test_default() {
            assert!(matches!(CellValue::default(), CellValue::Null));
        }

        #[test]
        fn test_null() {
            let (rendered, _) = CellValue::Null.render(5);
            assert!(rendered.is_empty());
        }

        #[test]
        fn test_bool_true() {
            let (rendered, _) = CellValue::Bool(true).render(5);
            assert_eq!(rendered, "true");
        }

        #[test]
        fn test_bool_false() {
            let (rendered, _) = CellValue::Bool(false).render(5);
            assert_eq!(rendered, "false");
        }

        #[test]
        fn test_int() {
            let (rendered, _) = CellValue::Int(42).render(5);
            assert_eq!(rendered, "42");
        }

        #[test]
        fn test_float() {
            let (rendered, _) = CellValue::Float(1.23456).render(10);
            assert!(rendered.starts_with("1.23"));
        }

        #[test]
        fn test_text() {
            let (rendered, _) = CellValue::Text("hello".to_string()).render(10);
            assert_eq!(rendered, "hello");
        }

        #[test]
        fn test_sparkline() {
            let (rendered, _) = CellValue::Sparkline(vec![1.0, 5.0, 3.0, 8.0, 2.0]).render(5);
            assert_eq!(rendered.chars().count(), 5);
        }

        #[test]
        fn test_sparkline_empty() {
            let (rendered, _) = CellValue::Sparkline(vec![]).render(5);
            assert_eq!(rendered.len(), 5);
        }

        #[test]
        fn test_sparkline_with_nan() {
            let (rendered, _) = CellValue::Sparkline(vec![1.0, f64::NAN, 3.0]).render(3);
            assert_eq!(rendered.chars().count(), 3);
        }

        #[test]
        fn test_progress() {
            let (rendered, _) = CellValue::Progress(50.0).render(15);
            assert!(rendered.contains("50%"));
        }

        #[test]
        fn test_progress_clamp() {
            let (rendered, _) = CellValue::Progress(150.0).render(15);
            assert!(rendered.contains("100%"));
        }

        #[test]
        fn test_status() {
            let (rendered, color) = CellValue::Status(StatusLevel::Ok).render(1);
            assert_eq!(rendered, "●");
            assert_eq!(color, Color::Green);
        }

        #[test]
        fn test_trend_up() {
            let (rendered, color) = CellValue::Trend(0.15).render(10);
            assert!(rendered.contains('↑'));
            assert_eq!(color, Color::Green);
        }

        #[test]
        fn test_trend_down() {
            let (rendered, color) = CellValue::Trend(-0.15).render(10);
            assert!(rendered.contains('↓'));
            assert_eq!(color, Color::Red);
        }

        #[test]
        fn test_trend_flat() {
            let (rendered, _) = CellValue::Trend(0.0).render(10);
            assert!(rendered.contains('→'));
        }

        #[test]
        fn test_microbar() {
            let (rendered, _) = CellValue::MicroBar { value: 5.0, max: 10.0 }.render(10);
            assert!(rendered.contains('█'));
            assert!(rendered.contains('░'));
        }

        #[test]
        fn test_microbar_full() {
            let (rendered, _) = CellValue::MicroBar { value: 10.0, max: 10.0 }.render(10);
            assert_eq!(rendered.chars().filter(|&c| c == '█').count(), 10);
        }
    }

    mod column_align_tests {
        use super::*;

        #[test]
        fn test_default() {
            assert_eq!(ColumnAlign::default(), ColumnAlign::Left);
        }

        #[test]
        fn test_equality() {
            assert_eq!(ColumnAlign::Right, ColumnAlign::Right);
            assert_ne!(ColumnAlign::Left, ColumnAlign::Right);
        }
    }

    mod column_tests {
        use super::*;

        #[test]
        fn test_new() {
            let col = Column::new("Test");
            assert_eq!(col.name, "Test");
            assert!(col.values.is_empty());
            assert_eq!(col.width, 10);
        }

        #[test]
        fn test_width() {
            let col = Column::new("Test").width(20);
            assert_eq!(col.width, 20);
        }

        #[test]
        fn test_width_minimum() {
            let col = Column::new("Test").width(1);
            assert_eq!(col.width, 3);
        }

        #[test]
        fn test_align() {
            let col = Column::new("Test").align(ColumnAlign::Right);
            assert_eq!(col.align, ColumnAlign::Right);
        }

        #[test]
        fn test_values() {
            let col = Column::new("Test").values(vec![CellValue::Int(1), CellValue::Int(2)]);
            assert_eq!(col.values.len(), 2);
        }

        #[test]
        fn test_from_f64() {
            let col = Column::from_f64("Numbers", &[1.0, 2.0, 3.0]);
            assert_eq!(col.values.len(), 3);
            assert_eq!(col.align, ColumnAlign::Right);
        }

        #[test]
        fn test_from_i64() {
            let col = Column::from_i64("Ints", &[1, 2, 3]);
            assert_eq!(col.values.len(), 3);
            assert_eq!(col.align, ColumnAlign::Right);
        }

        #[test]
        fn test_from_strings() {
            let col = Column::from_strings("Names", &["Alice", "Bob"]);
            assert_eq!(col.values.len(), 2);
            assert_eq!(col.align, ColumnAlign::Left);
        }

        #[test]
        fn test_sparklines() {
            let rows = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
            let col = Column::sparklines("Trend", rows);
            assert_eq!(col.values.len(), 2);
            assert_eq!(col.width, 12);
        }
    }

    mod dataframe_tests {
        use super::*;

        #[test]
        fn test_new() {
            let df = DataFrame::new();
            assert_eq!(df.row_count(), 0);
            assert_eq!(df.column_count(), 0);
        }

        #[test]
        fn test_default() {
            let df = DataFrame::default();
            assert!(df.columns.is_empty());
        }

        #[test]
        fn test_column() {
            let df = DataFrame::new()
                .column(Column::from_f64("A", &[1.0, 2.0, 3.0]))
                .column(Column::from_f64("B", &[4.0, 5.0, 6.0]));
            assert_eq!(df.row_count(), 3);
            assert_eq!(df.column_count(), 2);
        }

        #[test]
        fn test_visible_rows() {
            let df = DataFrame::new().visible_rows(50);
            assert_eq!(df.visible_rows, 50);
        }

        #[test]
        fn test_show_header() {
            let df = DataFrame::new().show_header(false);
            assert!(!df.show_header);
        }

        #[test]
        fn test_show_row_numbers() {
            let df = DataFrame::new().show_row_numbers(false);
            assert!(!df.show_row_numbers);
        }

        #[test]
        fn test_title() {
            let df = DataFrame::new().title("My Table");
            assert_eq!(df.title.as_deref(), Some("My Table"));
        }

        #[test]
        fn test_select_row() {
            let mut df = DataFrame::new().column(Column::from_f64("A", &[1.0, 2.0]));
            df.select_row(Some(1));
            assert_eq!(df.selected_row, Some(1));
            df.select_row(None);
            assert_eq!(df.selected_row, None);
        }

        #[test]
        fn test_scroll_to() {
            let mut df = DataFrame::new()
                .column(Column::from_f64("A", &(0..100).map(f64::from).collect::<Vec<_>>()))
                .visible_rows(10);
            df.scroll_to(50);
            assert_eq!(df.scroll_offset, 50);
        }

        #[test]
        fn test_scroll_to_beyond() {
            let mut df =
                DataFrame::new().column(Column::from_f64("A", &[1.0, 2.0, 3.0])).visible_rows(10);
            df.scroll_to(100);
            assert!(df.scroll_offset <= df.row_count());
        }

        #[test]
        fn test_builder_chaining() {
            let df = DataFrame::new()
                .column(Column::from_f64("A", &[1.0, 2.0]))
                .visible_rows(30)
                .show_header(true)
                .show_row_numbers(false)
                .title("Test");

            assert_eq!(df.column_count(), 1);
            assert_eq!(df.visible_rows, 30);
            assert!(df.show_header);
            assert!(!df.show_row_numbers);
        }
    }

    mod rendering_tests {
        use super::*;

        fn create_test_buffer(width: u16, height: u16) -> (Rect, Buffer) {
            let area = Rect::new(0, 0, width, height);
            let buf = Buffer::empty(area);
            (area, buf)
        }

        #[test]
        fn test_render_empty() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let df = DataFrame::new();
            df.render(area, &mut buf);
        }

        #[test]
        fn test_render_small_area() {
            let (area, mut buf) = create_test_buffer(5, 2);
            let df = DataFrame::new().column(Column::from_f64("A", &[1.0]));
            df.render(area, &mut buf);
        }

        #[test]
        fn test_render_basic() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let df = DataFrame::new()
                .column(Column::from_strings("Name", &["Alice", "Bob", "Carol"]))
                .column(Column::from_f64("Score", &[95.0, 87.0, 92.0]));
            df.render(area, &mut buf);
        }

        #[test]
        fn test_render_with_title() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let df =
                DataFrame::new().column(Column::from_f64("A", &[1.0, 2.0])).title("Test DataFrame");
            df.render(area, &mut buf);
        }

        #[test]
        fn test_render_no_header() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let df = DataFrame::new().column(Column::from_f64("A", &[1.0, 2.0])).show_header(false);
            df.render(area, &mut buf);
        }

        #[test]
        fn test_render_no_row_numbers() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let df =
                DataFrame::new().column(Column::from_f64("A", &[1.0, 2.0])).show_row_numbers(false);
            df.render(area, &mut buf);
        }

        #[test]
        fn test_render_with_selection() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let mut df = DataFrame::new().column(Column::from_f64("A", &[1.0, 2.0, 3.0]));
            df.select_row(Some(1));
            df.render(area, &mut buf);
        }

        #[test]
        fn test_render_all_cell_types() {
            let (area, mut buf) = create_test_buffer(80, 20);
            let df = DataFrame::new().column(
                Column::new("Types")
                    .values(vec![
                        CellValue::Null,
                        CellValue::Bool(true),
                        CellValue::Int(42),
                        CellValue::Float(1.23),
                        CellValue::Text("text".to_string()),
                        CellValue::Sparkline(vec![1.0, 2.0, 3.0]),
                        CellValue::Progress(75.0),
                        CellValue::Status(StatusLevel::Ok),
                        CellValue::Trend(0.05),
                        CellValue::MicroBar { value: 5.0, max: 10.0 },
                    ])
                    .width(20),
            );
            df.render(area, &mut buf);
        }

        #[test]
        fn test_render_with_scrolling() {
            let (area, mut buf) = create_test_buffer(60, 10);
            let values: Vec<f64> = (0..50).map(f64::from).collect();
            let mut df = DataFrame::new().column(Column::from_f64("A", &values)).visible_rows(5);
            df.scroll_to(20);
            df.render(area, &mut buf);
        }

        #[test]
        fn test_render_cell_alignment() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let df = DataFrame::new()
                .column(Column::from_strings("Left", &["a", "b"]).align(ColumnAlign::Left))
                .column(Column::from_f64("Right", &[1.0, 2.0]).align(ColumnAlign::Right))
                .column(
                    Column::new("Center")
                        .values(vec![
                            CellValue::Text("x".to_string()),
                            CellValue::Text("y".to_string()),
                        ])
                        .align(ColumnAlign::Center),
                );
            df.render(area, &mut buf);
        }

        #[test]
        fn test_render_very_narrow() {
            // Very narrow area - tests width boundary clipping
            let (area, mut buf) = create_test_buffer(10, 15);
            let df = DataFrame::new()
                .column(
                    Column::from_strings("Name", &["VeryLongName", "AnotherLongName"]).width(15),
                )
                .column(Column::from_f64("Score", &[95.0, 87.0]).width(10))
                .title("Test");
            df.render(area, &mut buf);
        }

        #[test]
        fn test_render_short_height() {
            // Short height - tests height boundary (y >= area.y + area.height break)
            let (area, mut buf) = create_test_buffer(60, 5);
            let values: Vec<f64> = (0..20).map(f64::from).collect();
            let df = DataFrame::new()
                .column(Column::from_f64("A", &values))
                .title("Title")
                .visible_rows(20);
            df.render(area, &mut buf);
        }

        #[test]
        fn test_render_with_trend_variations() {
            let (area, mut buf) = create_test_buffer(60, 15);
            let df = DataFrame::new().column(
                Column::new("Trend")
                    .values(vec![
                        CellValue::Trend(0.15),  // Up (> 0.1)
                        CellValue::Trend(0.05),  // Slightly up (> 0.02, < 0.1)
                        CellValue::Trend(0.0),   // Flat (> -0.02, < 0.02)
                        CellValue::Trend(-0.05), // Slightly down (> -0.1, < -0.02)
                        CellValue::Trend(-0.15), // Down (< -0.1)
                    ])
                    .width(12),
            );
            df.render(area, &mut buf);
        }

        #[test]
        fn test_render_title_clipping() {
            let (area, mut buf) = create_test_buffer(10, 10);
            let df = DataFrame::new()
                .column(Column::from_f64("A", &[1.0]))
                .title("This Is A Very Long Title That Will Be Clipped");
            df.render(area, &mut buf);
        }

        #[test]
        fn test_render_header_clipping() {
            let (area, mut buf) = create_test_buffer(8, 10);
            let df = DataFrame::new().column(
                Column::new("VeryLongColumnName").values(vec![CellValue::Int(1)]).width(15),
            );
            df.render(area, &mut buf);
        }
    }
}
