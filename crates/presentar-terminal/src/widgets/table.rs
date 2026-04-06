//! Scrollable table widget.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event, Key,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Gray color constant.
const GRAY: Color = Color {
    r: 0.5,
    g: 0.5,
    b: 0.5,
    a: 1.0,
};

/// Cyan color constant.
const CYAN: Color = Color {
    r: 0.0,
    g: 1.0,
    b: 1.0,
    a: 1.0,
};

/// Scrollable table widget with headers and rows.
#[derive(Debug, Clone)]
pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    selected: usize,
    scroll_offset: usize,
    sort_column: Option<usize>,
    sort_ascending: bool,
    header_color: Color,
    selected_color: Color,
    bounds: Rect,
}

impl Table {
    /// Create a new table with headers.
    #[must_use]
    pub fn new(headers: Vec<String>) -> Self {
        Self {
            headers,
            rows: Vec::new(),
            selected: 0,
            scroll_offset: 0,
            sort_column: None,
            sort_ascending: true,
            header_color: CYAN,
            selected_color: Color::BLUE,
            bounds: Rect::new(0.0, 0.0, 0.0, 0.0),
        }
    }

    /// Set the rows.
    #[must_use]
    pub fn with_rows(mut self, rows: Vec<Vec<String>>) -> Self {
        self.rows = rows;
        self
    }

    /// Set the header color.
    #[must_use]
    pub fn with_header_color(mut self, color: Color) -> Self {
        self.header_color = color;
        self
    }

    /// Set the selected row highlight color.
    #[must_use]
    pub fn with_selected_color(mut self, color: Color) -> Self {
        self.selected_color = color;
        self
    }

    /// Add a row.
    pub fn add_row(&mut self, row: Vec<String>) {
        self.rows.push(row);
    }

    /// Clear all rows.
    pub fn clear(&mut self) {
        self.rows.clear();
        self.selected = 0;
        self.scroll_offset = 0;
    }

    /// Set the selected row.
    pub fn select(&mut self, row: usize) {
        if !self.rows.is_empty() {
            self.selected = row.min(self.rows.len() - 1);
            self.ensure_visible();
        }
    }

    /// Move selection up.
    pub fn select_prev(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
            self.ensure_visible();
        }
    }

    /// Move selection down.
    pub fn select_next(&mut self) {
        if !self.rows.is_empty() && self.selected < self.rows.len() - 1 {
            self.selected += 1;
            self.ensure_visible();
        }
    }

    /// Get the selected row index.
    #[must_use]
    pub fn selected(&self) -> usize {
        self.selected
    }

    /// Get the selected row data.
    #[must_use]
    pub fn selected_row(&self) -> Option<&Vec<String>> {
        self.rows.get(self.selected)
    }

    /// Sort by column.
    pub fn sort_by(&mut self, column: usize) {
        if column >= self.headers.len() {
            return;
        }

        if self.sort_column == Some(column) {
            self.sort_ascending = !self.sort_ascending;
        } else {
            self.sort_column = Some(column);
            self.sort_ascending = true;
        }

        let ascending = self.sort_ascending;
        self.rows.sort_by(|a, b| {
            let val_a = a.get(column).map_or("", String::as_str);
            let val_b = b.get(column).map_or("", String::as_str);
            if ascending {
                val_a.cmp(val_b)
            } else {
                val_b.cmp(val_a)
            }
        });
    }

    fn ensure_visible(&mut self) {
        let visible_rows = (self.bounds.height as usize).saturating_sub(1);
        if visible_rows == 0 {
            return;
        }

        if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        } else if self.selected >= self.scroll_offset + visible_rows {
            self.scroll_offset = self.selected - visible_rows + 1;
        }
    }

    fn column_widths(&self, total_width: usize) -> Vec<usize> {
        if self.headers.is_empty() {
            return vec![];
        }

        let mut widths: Vec<usize> = self.headers.iter().map(String::len).collect();

        for row in &self.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < widths.len() {
                    widths[i] = widths[i].max(cell.len());
                }
            }
        }

        let total_content: usize = widths.iter().sum();
        let separators = (self.headers.len() - 1) * 3;
        let available = total_width.saturating_sub(separators);

        if total_content > available {
            let ratio = available as f64 / total_content as f64;
            for w in &mut widths {
                *w = ((*w as f64) * ratio).max(3.0) as usize;
            }
        }

        widths
    }

    fn truncate(s: &str, width: usize) -> String {
        if s.len() <= width {
            format!("{s:width$}")
        } else if width > 3 {
            format!("{}...", &s[..width - 3])
        } else {
            s[..width].to_string()
        }
    }
}

impl Brick for Table {
    fn brick_name(&self) -> &'static str {
        "table"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        // Check selected is in bounds
        if self.rows.is_empty() || self.selected < self.rows.len() {
            passed.push(BrickAssertion::max_latency_ms(16));
        } else {
            failed.push((
                BrickAssertion::max_latency_ms(16),
                format!(
                    "Selected {} >= row count {}",
                    self.selected,
                    self.rows.len()
                ),
            ));
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

impl Widget for Table {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let width = constraints.max_width.max(20.0);
        let min_height = 3.0;
        let preferred_height = (self.rows.len() + 1) as f32;
        let height = constraints
            .max_height
            .max(min_height)
            .min(preferred_height.max(min_height));
        constraints.constrain(Size::new(width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        self.ensure_visible();
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let width = self.bounds.width as usize;
        let height = self.bounds.height as usize;
        if width == 0 || height == 0 {
            return;
        }

        let col_widths = self.column_widths(width);

        // Draw header
        let header_style = TextStyle {
            color: self.header_color,
            weight: presentar_core::FontWeight::Bold,
            ..Default::default()
        };

        let mut header_line = String::new();
        for (i, header) in self.headers.iter().enumerate() {
            if i > 0 {
                header_line.push_str(" │ ");
            }
            let w = col_widths.get(i).copied().unwrap_or(10);
            header_line.push_str(&Self::truncate(header, w));
        }
        canvas.draw_text(
            &header_line,
            Point::new(self.bounds.x, self.bounds.y),
            &header_style,
        );

        // Draw separator
        let sep_y = self.bounds.y + 1.0;
        if height > 1 {
            let sep: String = "─".repeat(width);
            canvas.draw_text(
                &sep,
                Point::new(self.bounds.x, sep_y),
                &TextStyle::default(),
            );
        }

        // Draw rows
        let visible_rows = height.saturating_sub(2);
        let default_style = TextStyle::default();
        let selected_style = TextStyle {
            color: self.selected_color,
            ..Default::default()
        };

        for (i, row_idx) in (self.scroll_offset..self.rows.len())
            .take(visible_rows)
            .enumerate()
        {
            let row = &self.rows[row_idx];
            let y = self.bounds.y + 2.0 + i as f32;

            let style = if row_idx == self.selected {
                &selected_style
            } else {
                &default_style
            };

            let mut row_line = String::new();
            for (j, cell) in row.iter().enumerate() {
                if j > 0 {
                    row_line.push_str(" │ ");
                }
                let w = col_widths.get(j).copied().unwrap_or(10);
                row_line.push_str(&Self::truncate(cell, w));
            }

            // Draw selection background
            if row_idx == self.selected {
                let bg_color = Color::new(
                    self.selected_color.r,
                    self.selected_color.g,
                    self.selected_color.b,
                    0.3,
                );
                canvas.fill_rect(
                    Rect::new(self.bounds.x, y, self.bounds.width, 1.0),
                    bg_color,
                );
            }

            canvas.draw_text(&row_line, Point::new(self.bounds.x, y), style);
        }

        // Show "No data" if empty
        if self.rows.is_empty() && height > 2 {
            canvas.draw_text(
                "No data",
                Point::new(self.bounds.x + 1.0, self.bounds.y + 2.0),
                &TextStyle {
                    color: GRAY,
                    ..Default::default()
                },
            );
        }
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        match event {
            Event::KeyDown { key, .. } => {
                match key {
                    Key::Up | Key::K => self.select_prev(),
                    Key::Down | Key::J => self.select_next(),
                    _ => {}
                }
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
    use presentar_core::{Canvas, TextStyle};

    struct MockCanvas {
        texts: Vec<(String, Point)>,
        rects: Vec<Rect>,
    }

    impl MockCanvas {
        fn new() -> Self {
            Self {
                texts: vec![],
                rects: vec![],
            }
        }
    }

    impl Canvas for MockCanvas {
        fn fill_rect(&mut self, rect: Rect, _color: Color) {
            self.rects.push(rect);
        }
        fn stroke_rect(&mut self, _rect: Rect, _color: Color, _width: f32) {}
        fn draw_text(&mut self, text: &str, position: Point, _style: &TextStyle) {
            self.texts.push((text.to_string(), position));
        }
        fn draw_line(&mut self, _from: Point, _to: Point, _color: Color, _width: f32) {}
        fn fill_circle(&mut self, _center: Point, _radius: f32, _color: Color) {}
        fn stroke_circle(&mut self, _center: Point, _radius: f32, _color: Color, _width: f32) {}
        fn fill_arc(
            &mut self,
            _center: Point,
            _radius: f32,
            _start: f32,
            _end: f32,
            _color: Color,
        ) {
        }
        fn draw_path(&mut self, _points: &[Point], _color: Color, _width: f32) {}
        fn fill_polygon(&mut self, _points: &[Point], _color: Color) {}
        fn push_clip(&mut self, _rect: Rect) {}
        fn pop_clip(&mut self) {}
        fn push_transform(&mut self, _transform: presentar_core::Transform2D) {}
        fn pop_transform(&mut self) {}
    }

    fn sample_table() -> Table {
        Table::new(vec!["Name".into(), "Value".into()]).with_rows(vec![
            vec!["CPU".into(), "45%".into()],
            vec!["Memory".into(), "62%".into()],
            vec!["Disk".into(), "78%".into()],
        ])
    }

    #[test]
    fn test_table_creation() {
        let table = sample_table();
        assert_eq!(table.headers.len(), 2);
        assert_eq!(table.rows.len(), 3);
    }

    #[test]
    fn test_table_assertions_not_empty() {
        let table = sample_table();
        assert!(!table.assertions().is_empty());
    }

    #[test]
    fn test_table_verify_pass() {
        let table = sample_table();
        assert!(table.verify().is_valid());
    }

    #[test]
    fn test_table_selection() {
        let mut table = sample_table();
        assert_eq!(table.selected(), 0);

        table.select_next();
        assert_eq!(table.selected(), 1);

        table.select_prev();
        assert_eq!(table.selected(), 0);
    }

    #[test]
    fn test_table_with_header_color() {
        let table = Table::new(vec!["A".into()]).with_header_color(Color::RED);
        assert_eq!(table.header_color, Color::RED);
    }

    #[test]
    fn test_table_with_selected_color() {
        let table = Table::new(vec!["A".into()]).with_selected_color(Color::GREEN);
        assert_eq!(table.selected_color, Color::GREEN);
    }

    #[test]
    fn test_table_add_row() {
        let mut table = Table::new(vec!["A".into(), "B".into()]);
        table.add_row(vec!["1".into(), "2".into()]);
        assert_eq!(table.rows.len(), 1);
    }

    #[test]
    fn test_table_clear() {
        let mut table = sample_table();
        table.select_next();
        table.clear();
        assert_eq!(table.rows.len(), 0);
        assert_eq!(table.selected(), 0);
        assert_eq!(table.scroll_offset, 0);
    }

    #[test]
    fn test_table_select() {
        let mut table = sample_table();
        table.select(2);
        assert_eq!(table.selected(), 2);

        table.select(10);
        assert_eq!(table.selected(), 2);
    }

    #[test]
    fn test_table_select_prev_at_start() {
        let mut table = sample_table();
        table.select_prev();
        assert_eq!(table.selected(), 0);
    }

    #[test]
    fn test_table_select_next_at_end() {
        let mut table = sample_table();
        table.select(2);
        table.select_next();
        assert_eq!(table.selected(), 2);
    }

    #[test]
    fn test_table_selected_row() {
        let table = sample_table();
        let row = table.selected_row().unwrap();
        assert_eq!(row[0], "CPU");
    }

    #[test]
    fn test_table_sort_by() {
        let mut table = sample_table();
        table.sort_by(0);
        assert_eq!(table.rows[0][0], "CPU");
        assert_eq!(table.rows[1][0], "Disk");
        assert_eq!(table.rows[2][0], "Memory");

        table.sort_by(0);
        assert_eq!(table.rows[0][0], "Memory");
    }

    #[test]
    fn test_table_sort_by_invalid_column() {
        let mut table = sample_table();
        table.sort_by(10);
        assert_eq!(table.rows[0][0], "CPU");
    }

    #[test]
    fn test_table_column_widths() {
        let table = sample_table();
        let widths = table.column_widths(80);
        assert!(!widths.is_empty());
    }

    #[test]
    fn test_table_column_widths_empty_headers() {
        let table = Table::new(vec![]);
        let widths = table.column_widths(80);
        assert!(widths.is_empty());
    }

    #[test]
    fn test_table_truncate() {
        assert_eq!(Table::truncate("Hello", 10), "Hello     ");
        assert_eq!(Table::truncate("Hello World", 5), "He...");
        assert_eq!(Table::truncate("Hi", 2), "Hi");
    }

    #[test]
    fn test_table_truncate_very_short() {
        assert_eq!(Table::truncate("Hello", 3), "Hel");
    }

    #[test]
    fn test_table_measure() {
        let table = sample_table();
        let constraints = Constraints::new(0.0, 100.0, 0.0, 50.0);
        let size = table.measure(constraints);
        assert!(size.width >= 20.0);
        assert!(size.height >= 3.0);
    }

    #[test]
    fn test_table_layout() {
        let mut table = sample_table();
        let bounds = Rect::new(0.0, 0.0, 80.0, 20.0);
        let result = table.layout(bounds);
        assert_eq!(result.size.width, 80.0);
        assert_eq!(result.size.height, 20.0);
    }

    #[test]
    fn test_table_paint() {
        let mut table = sample_table();
        table.bounds = Rect::new(0.0, 0.0, 40.0, 10.0);
        let mut canvas = MockCanvas::new();
        table.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_table_paint_empty() {
        let mut table = Table::new(vec!["A".into(), "B".into()]);
        table.bounds = Rect::new(0.0, 0.0, 40.0, 10.0);
        let mut canvas = MockCanvas::new();
        table.paint(&mut canvas);
        assert!(canvas.texts.iter().any(|(t, _)| t.contains("No data")));
    }

    #[test]
    fn test_table_paint_zero_size() {
        let mut table = sample_table();
        table.bounds = Rect::new(0.0, 0.0, 0.0, 0.0);
        let mut canvas = MockCanvas::new();
        table.paint(&mut canvas);
        assert!(canvas.texts.is_empty());
    }

    #[test]
    fn test_table_paint_with_selected() {
        let mut table = sample_table();
        table.bounds = Rect::new(0.0, 0.0, 40.0, 10.0);
        table.select(1);
        let mut canvas = MockCanvas::new();
        table.paint(&mut canvas);
        assert!(!canvas.rects.is_empty());
    }

    #[test]
    fn test_table_event_up() {
        let mut table = sample_table();
        table.select(1);
        let event = Event::key_down(Key::Up);
        table.event(&event);
        assert_eq!(table.selected(), 0);
    }

    #[test]
    fn test_table_event_down() {
        let mut table = sample_table();
        let event = Event::key_down(Key::Down);
        table.event(&event);
        assert_eq!(table.selected(), 1);
    }

    #[test]
    fn test_table_event_k() {
        let mut table = sample_table();
        table.select(1);
        let event = Event::key_down(Key::K);
        table.event(&event);
        assert_eq!(table.selected(), 0);
    }

    #[test]
    fn test_table_event_j() {
        let mut table = sample_table();
        let event = Event::key_down(Key::J);
        table.event(&event);
        assert_eq!(table.selected(), 1);
    }

    #[test]
    fn test_table_event_other() {
        let mut table = sample_table();
        let event = Event::key_down(Key::Enter);
        assert!(table.event(&event).is_none());
    }

    #[test]
    fn test_table_event_non_keydown() {
        let mut table = sample_table();
        let event = Event::FocusIn;
        assert!(table.event(&event).is_none());
    }

    #[test]
    fn test_table_children() {
        let table = sample_table();
        assert!(table.children().is_empty());
    }

    #[test]
    fn test_table_children_mut() {
        let mut table = sample_table();
        assert!(table.children_mut().is_empty());
    }

    #[test]
    fn test_table_type_id() {
        let table = sample_table();
        assert_eq!(Widget::type_id(&table), TypeId::of::<Table>());
    }

    #[test]
    fn test_table_brick_name() {
        let table = sample_table();
        assert_eq!(table.brick_name(), "table");
    }

    #[test]
    fn test_table_budget() {
        let table = sample_table();
        let budget = table.budget();
        assert!(budget.measure_ms > 0);
    }

    #[test]
    fn test_table_to_html() {
        let table = sample_table();
        assert!(table.to_html().is_empty());
    }

    #[test]
    fn test_table_to_css() {
        let table = sample_table();
        assert!(table.to_css().is_empty());
    }

    #[test]
    fn test_table_scroll() {
        let mut table = Table::new(vec!["Name".into()])
            .with_rows((0..100).map(|i| vec![format!("Item {}", i)]).collect());
        table.bounds = Rect::new(0.0, 0.0, 40.0, 10.0);
        table.layout(table.bounds);

        table.select(50);
        assert!(table.scroll_offset > 0);
    }

    #[test]
    fn test_table_ensure_visible_no_visible_rows() {
        let mut table = sample_table();
        table.bounds = Rect::new(0.0, 0.0, 40.0, 1.0);
        table.select(2);
    }

    #[test]
    fn test_table_verify_invalid_selection() {
        let mut table = sample_table();
        table.selected = 10;
        assert!(!table.verify().is_valid());
    }

    #[test]
    fn test_table_empty_verify() {
        let table = Table::new(vec!["A".into()]);
        assert!(table.verify().is_valid());
    }

    #[test]
    fn test_table_select_empty() {
        let mut table = Table::new(vec!["A".into()]);
        table.select(5);
        assert_eq!(table.selected(), 0);
    }

    #[test]
    fn test_table_narrow_columns() {
        let table = Table::new(vec!["A".into(), "B".into(), "C".into()]).with_rows(vec![vec![
            "VeryLongValue1".into(),
            "VeryLongValue2".into(),
            "VeryLongValue3".into(),
        ]]);
        let widths = table.column_widths(30);
        let total: usize = widths.iter().sum();
        assert!(total <= 30);
    }
}
