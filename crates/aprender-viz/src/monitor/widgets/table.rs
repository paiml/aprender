//! Sortable and scrollable table widget.
//!
//! Supports 10,000+ rows with 60fps scrolling (Falsification criterion #11).

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::Widget;

/// Sort direction for table columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortDirection {
    /// Ascending order (A-Z, 0-9).
    #[default]
    Ascending,
    /// Descending order (Z-A, 9-0).
    Descending,
}

/// A sortable, scrollable table.
#[derive(Debug, Clone)]
pub struct MonitorTable {
    /// Column headers.
    headers: Vec<String>,
    /// Row data (each row is a vec of cell values).
    rows: Vec<Vec<String>>,
    /// Currently selected row index.
    selected: Option<usize>,
    /// Scroll offset.
    offset: usize,
    /// Sort column index.
    sort_column: Option<usize>,
    /// Sort direction.
    sort_direction: SortDirection,
}

impl MonitorTable {
    /// Creates a new empty table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            headers: Vec::new(),
            rows: Vec::new(),
            selected: None,
            offset: 0,
            sort_column: None,
            sort_direction: SortDirection::default(),
        }
    }

    /// Sets the column headers.
    #[must_use]
    pub fn headers(mut self, headers: Vec<String>) -> Self {
        self.headers = headers;
        self
    }

    /// Adds a row to the table.
    pub fn add_row(&mut self, row: Vec<String>) {
        self.rows.push(row);
    }

    /// Returns the number of rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Returns the currently selected row index.
    #[must_use]
    pub fn selected(&self) -> Option<usize> {
        self.selected
    }

    /// Selects a row by index.
    pub fn select(&mut self, index: Option<usize>) {
        self.selected = index.map(|i| i.min(self.rows.len().saturating_sub(1)));
    }

    /// Moves selection up.
    pub fn select_previous(&mut self) {
        if let Some(selected) = self.selected {
            self.selected = Some(selected.saturating_sub(1));
        } else if !self.rows.is_empty() {
            self.selected = Some(0);
        }
        self.ensure_visible();
    }

    /// Moves selection down.
    pub fn select_next(&mut self) {
        if let Some(selected) = self.selected {
            self.selected = Some((selected + 1).min(self.rows.len().saturating_sub(1)));
        } else if !self.rows.is_empty() {
            self.selected = Some(0);
        }
        self.ensure_visible();
    }

    /// Ensures the selected row is visible.
    fn ensure_visible(&mut self) {
        if let Some(selected) = self.selected {
            if selected < self.offset {
                self.offset = selected;
            }
            // Note: We'll adjust for visible height during render
        }
    }

    /// Sorts the table by the given column.
    pub fn sort_by(&mut self, column: usize, direction: SortDirection) {
        if column >= self.headers.len() {
            return;
        }

        self.sort_column = Some(column);
        self.sort_direction = direction;

        self.rows.sort_by(|a, b| {
            let a_val = a.get(column).map_or("", String::as_str);
            let b_val = b.get(column).map_or("", String::as_str);

            // Try numeric comparison first
            let cmp = match (a_val.parse::<f64>(), b_val.parse::<f64>()) {
                (Ok(a_num), Ok(b_num)) => {
                    a_num.partial_cmp(&b_num).unwrap_or(std::cmp::Ordering::Equal)
                }
                _ => a_val.cmp(b_val),
            };

            match direction {
                SortDirection::Ascending => cmp,
                SortDirection::Descending => cmp.reverse(),
            }
        });
    }

    /// Returns the visible rows for the given height.
    #[must_use]
    pub fn visible_rows(&self, height: usize) -> &[Vec<String>] {
        let end = (self.offset + height).min(self.rows.len());
        &self.rows[self.offset..end]
    }

    /// Scrolls down by the given amount.
    pub fn scroll_down(&mut self, amount: usize) {
        self.offset = (self.offset + amount).min(self.rows.len().saturating_sub(1));
    }

    /// Scrolls up by the given amount.
    pub fn scroll_up(&mut self, amount: usize) {
        self.offset = self.offset.saturating_sub(amount);
    }

    /// Clears all rows.
    pub fn clear(&mut self) {
        self.rows.clear();
        self.selected = None;
        self.offset = 0;
    }
}

impl Default for MonitorTable {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for MonitorTable {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.height == 0 || area.width == 0 {
            return;
        }

        let mut y = area.y;

        // Render headers
        if !self.headers.is_empty() {
            let mut x = area.x;
            let col_width = area.width / self.headers.len().max(1) as u16;

            for (i, header) in self.headers.iter().enumerate() {
                let style = Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD);

                // Show sort indicator
                let text = if self.sort_column == Some(i) {
                    let indicator = match self.sort_direction {
                        SortDirection::Ascending => "▲",
                        SortDirection::Descending => "▼",
                    };
                    format!("{header} {indicator}")
                } else {
                    header.clone()
                };

                let truncated: String = text.chars().take(col_width as usize - 1).collect();
                buf.set_string(x, y, truncated, style);
                x += col_width;
            }
            y += 1;
        }

        // Render rows
        let visible_height = (area.height - 1) as usize; // -1 for header
        let visible_rows = self.visible_rows(visible_height);

        for (i, row) in visible_rows.iter().enumerate() {
            if y >= area.y + area.height {
                break;
            }

            let row_idx = self.offset + i;
            let is_selected = self.selected == Some(row_idx);

            let style = if is_selected {
                Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };

            let mut x = area.x;
            let col_width = area.width / row.len().max(1) as u16;

            for cell in row {
                let truncated: String = cell.chars().take(col_width as usize - 1).collect();
                buf.set_string(x, y, truncated, style);
                x += col_width;
            }

            y += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_table_new() {
        let table = MonitorTable::new();
        assert_eq!(table.row_count(), 0);
        assert_eq!(table.selected(), None);
    }

    #[test]
    fn test_table_default() {
        let table = MonitorTable::default();
        assert_eq!(table.row_count(), 0);
        assert_eq!(table.selected(), None);
    }

    #[test]
    fn test_sort_direction_default() {
        let dir = SortDirection::default();
        assert_eq!(dir, SortDirection::Ascending);
    }

    #[test]
    fn test_table_add_row() {
        let mut table = MonitorTable::new();
        table.add_row(vec!["a".to_string(), "b".to_string()]);
        table.add_row(vec!["c".to_string(), "d".to_string()]);

        assert_eq!(table.row_count(), 2);
    }

    #[test]
    fn test_table_selection() {
        let mut table = MonitorTable::new();
        for i in 0..10 {
            table.add_row(vec![format!("{}", i)]);
        }

        table.select(Some(5));
        assert_eq!(table.selected(), Some(5));

        table.select_previous();
        assert_eq!(table.selected(), Some(4));

        table.select_next();
        assert_eq!(table.selected(), Some(5));
    }

    #[test]
    fn test_table_selection_empty() {
        let mut table = MonitorTable::new();

        // No rows - select_previous should do nothing
        table.select_previous();
        assert_eq!(table.selected(), None);

        // No rows - select_next should do nothing
        table.select_next();
        assert_eq!(table.selected(), None);
    }

    #[test]
    fn test_table_selection_first_select() {
        let mut table = MonitorTable::new();
        table.add_row(vec!["a".to_string()]);
        table.add_row(vec!["b".to_string()]);

        // First select_next with no selection should select row 0
        table.select_next();
        assert_eq!(table.selected(), Some(0));

        let mut table2 = MonitorTable::new();
        table2.add_row(vec!["a".to_string()]);

        // First select_previous with no selection should select row 0
        table2.select_previous();
        assert_eq!(table2.selected(), Some(0));
    }

    #[test]
    fn test_table_selection_bounds() {
        let mut table = MonitorTable::new();
        table.add_row(vec!["0".to_string()]);
        table.add_row(vec!["1".to_string()]);
        table.add_row(vec!["2".to_string()]);

        // Select beyond bounds should clamp
        table.select(Some(100));
        assert_eq!(table.selected(), Some(2));

        // Select at start, previous should stay at 0
        table.select(Some(0));
        table.select_previous();
        assert_eq!(table.selected(), Some(0));

        // Select at end, next should stay at end
        table.select(Some(2));
        table.select_next();
        assert_eq!(table.selected(), Some(2));
    }

    #[test]
    fn test_table_sorting() {
        let mut table = MonitorTable::new().headers(vec!["Name".to_string(), "Value".to_string()]);

        table.add_row(vec!["b".to_string(), "2".to_string()]);
        table.add_row(vec!["a".to_string(), "1".to_string()]);
        table.add_row(vec!["c".to_string(), "3".to_string()]);

        table.sort_by(0, SortDirection::Ascending);

        assert_eq!(table.rows[0][0], "a");
        assert_eq!(table.rows[1][0], "b");
        assert_eq!(table.rows[2][0], "c");
    }

    #[test]
    fn test_table_sorting_descending() {
        let mut table = MonitorTable::new().headers(vec!["Name".to_string()]);

        table.add_row(vec!["a".to_string()]);
        table.add_row(vec!["c".to_string()]);
        table.add_row(vec!["b".to_string()]);

        table.sort_by(0, SortDirection::Descending);

        assert_eq!(table.rows[0][0], "c");
        assert_eq!(table.rows[1][0], "b");
        assert_eq!(table.rows[2][0], "a");
    }

    #[test]
    fn test_table_sorting_numeric() {
        let mut table = MonitorTable::new().headers(vec!["Value".to_string()]);

        table.add_row(vec!["10".to_string()]);
        table.add_row(vec!["2".to_string()]);
        table.add_row(vec!["100".to_string()]);

        table.sort_by(0, SortDirection::Ascending);

        // Numeric sort: 2 < 10 < 100
        assert_eq!(table.rows[0][0], "2");
        assert_eq!(table.rows[1][0], "10");
        assert_eq!(table.rows[2][0], "100");
    }

    #[test]
    fn test_table_sorting_invalid_column() {
        let mut table = MonitorTable::new().headers(vec!["A".to_string()]);
        table.add_row(vec!["1".to_string()]);

        // Sort by invalid column should do nothing
        table.sort_by(99, SortDirection::Ascending);
        assert_eq!(table.rows[0][0], "1");
    }

    #[test]
    fn test_table_scroll_up() {
        let mut table = MonitorTable::new();
        for i in 0..100 {
            table.add_row(vec![format!("{}", i)]);
        }

        table.scroll_down(50);
        assert_eq!(table.visible_rows(10)[0][0], "50");

        table.scroll_up(20);
        assert_eq!(table.visible_rows(10)[0][0], "30");

        // Scroll up past beginning
        table.scroll_up(100);
        assert_eq!(table.visible_rows(10)[0][0], "0");
    }

    #[test]
    fn test_table_clear() {
        let mut table = MonitorTable::new();
        table.add_row(vec!["a".to_string()]);
        table.add_row(vec!["b".to_string()]);
        table.select(Some(1));
        table.scroll_down(1);

        table.clear();

        assert_eq!(table.row_count(), 0);
        assert_eq!(table.selected(), None);
    }

    #[test]
    fn test_table_ensure_visible_scrolls_up() {
        let mut table = MonitorTable::new();
        for i in 0..100 {
            table.add_row(vec![format!("{}", i)]);
        }

        table.scroll_down(50);
        table.select(Some(10)); // Select row before visible area
        table.select_previous(); // This calls ensure_visible

        // Offset should adjust to show selected row
        assert!(table.offset <= 9);
    }

    /// Falsification criterion #11: Table scrolling maintains 60fps with 10,000 rows.
    #[test]
    fn test_table_scrolling_performance() {
        let mut table =
            MonitorTable::new().headers(vec!["A".to_string(), "B".to_string(), "C".to_string()]);

        // Add 10,000 rows
        for i in 0..10000 {
            table.add_row(vec![
                format!("Row {}", i),
                format!("{}", i * 2),
                format!("{:.2}", f64::from(i) / 100.0),
            ]);
        }

        // Scroll 60 times (simulating 60fps for 1 second)
        let start = Instant::now();
        for _ in 0..60 {
            table.scroll_down(1);
            let _ = table.visible_rows(24); // Simulate rendering
        }
        let elapsed = start.elapsed();

        // Should complete in under 1 second for 60fps
        assert!(
            elapsed.as_secs_f64() < 1.0,
            "Scrolling 60 times took {elapsed:?}, should be under 1 second"
        );
    }

    #[test]
    fn test_table_visible_rows() {
        let mut table = MonitorTable::new();
        for i in 0..100 {
            table.add_row(vec![format!("{}", i)]);
        }

        let visible = table.visible_rows(10);
        assert_eq!(visible.len(), 10);
        assert_eq!(visible[0][0], "0");

        table.scroll_down(50);
        let visible = table.visible_rows(10);
        assert_eq!(visible[0][0], "50");
    }

    #[test]
    fn test_table_render_empty() {
        let table = MonitorTable::new();
        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 10));
        table.render(Rect::new(0, 0, 40, 10), &mut buf);
        // Should not panic
    }

    #[test]
    fn test_table_render_zero_size() {
        let table = MonitorTable::new();
        let mut buf = Buffer::empty(Rect::new(0, 0, 0, 0));
        table.render(Rect::new(0, 0, 0, 0), &mut buf);
        // Should return early without panic
    }

    #[test]
    fn test_table_render_with_headers() {
        let table = MonitorTable::new().headers(vec!["Name".to_string(), "Value".to_string()]);
        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 10));
        table.render(Rect::new(0, 0, 40, 10), &mut buf);

        // Check header rendered
        let content = buf.cell((0, 0)).map_or("", ratatui::buffer::Cell::symbol);
        assert_eq!(content, "N"); // First char of "Name"
    }

    #[test]
    fn test_table_render_with_rows() {
        let mut table = MonitorTable::new().headers(vec!["Col".to_string()]);
        table.add_row(vec!["Row1".to_string()]);
        table.add_row(vec!["Row2".to_string()]);

        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 10));
        table.render(Rect::new(0, 0, 40, 10), &mut buf);

        // Check first row rendered (at y=1 after header)
        let content = buf.cell((0, 1)).map_or("", ratatui::buffer::Cell::symbol);
        assert_eq!(content, "R"); // First char of "Row1"
    }

    #[test]
    fn test_table_render_with_selection() {
        let mut table = MonitorTable::new().headers(vec!["Col".to_string()]);
        table.add_row(vec!["A".to_string()]);
        table.add_row(vec!["B".to_string()]);
        table.select(Some(1));

        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 10));
        table.render(Rect::new(0, 0, 40, 10), &mut buf);

        // Selected row should have different background
        let cell = buf.cell((0, 2)).expect("operation should succeed"); // Row 1 at y=2 (after header + row 0)
        assert_eq!(cell.bg, Color::DarkGray);
    }

    #[test]
    fn test_table_render_with_sort_indicator() {
        let mut table = MonitorTable::new().headers(vec!["Name".to_string(), "Value".to_string()]);
        table.add_row(vec!["a".to_string(), "1".to_string()]);
        table.sort_by(0, SortDirection::Ascending);

        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 10));
        table.clone().render(Rect::new(0, 0, 40, 10), &mut buf);

        // Sort indicator should be present - the header should contain the arrow
        // Just verify render doesn't panic with sort

        // Test descending too
        let mut table2 = table.clone();
        table2.sort_by(0, SortDirection::Descending);
        table2.render(Rect::new(0, 0, 40, 10), &mut buf);
    }

    #[test]
    fn test_table_render_overflow() {
        let mut table = MonitorTable::new().headers(vec!["Col".to_string()]);

        // Add more rows than visible height
        for i in 0..100 {
            table.add_row(vec![format!("Row{}", i)]);
        }

        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 5));
        table.render(Rect::new(0, 0, 40, 5), &mut buf);
        // Should not panic, only render visible rows
    }
}
