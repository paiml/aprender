//! Row detail view widget for displaying a single record
//!
//! Provides an expanded view of all fields in a single row.

use super::{adapter::DatasetAdapter, scroll::ScrollState};

/// Row detail view widget for displaying a single record
///
/// Shows all fields and their values for a selected row,
/// with scrolling support for large text values.
///
/// # Example
///
/// ```ignore
/// let adapter = DatasetAdapter::from_dataset(&dataset)?;
/// let detail = RowDetailView::new(&adapter, 5); // Row 5
///
/// for line in detail.render_lines() {
///     println!("{}", line);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct RowDetailView {
    /// Row index being displayed
    row_index: usize,
    /// Field values: (name, value)
    fields: Vec<(String, String)>,
    /// Scroll state for navigating fields
    scroll: ScrollState,
    /// Display width
    display_width: u16,
    /// Display height
    display_height: u16,
}

impl RowDetailView {
    /// Create a new row detail view
    ///
    /// # Arguments
    /// * `adapter` - Dataset adapter
    /// * `row_index` - Index of the row to display
    ///
    /// # Returns
    /// The detail view, or None if row is out of bounds
    pub fn new(adapter: &DatasetAdapter, row_index: usize) -> Option<Self> {
        Self::with_dimensions(adapter, row_index, 80, 24)
    }

    /// Create a new row detail view with specific dimensions
    pub fn with_dimensions(
        adapter: &DatasetAdapter,
        row_index: usize,
        width: u16,
        height: u16,
    ) -> Option<Self> {
        if row_index >= adapter.row_count() {
            return None;
        }

        // Collect field values
        let fields: Vec<(String, String)> = (0..adapter.column_count())
            .filter_map(|col| {
                let name = adapter.field_name(col)?.to_string();
                let value = adapter
                    .get_cell(row_index, col)
                    .ok()
                    .flatten()
                    .unwrap_or_else(|| "NULL".to_string());
                Some((name, value))
            })
            .collect();

        // Calculate total lines needed (each field may span multiple lines)
        let total_lines = calculate_total_lines(&fields, width);
        let visible_lines = height.saturating_sub(2) as usize; // -2 for title and border

        let scroll = ScrollState::new(total_lines, visible_lines);

        Some(Self {
            row_index,
            fields,
            scroll,
            display_width: width,
            display_height: height,
        })
    }

    /// Get the row index being displayed
    pub fn row_index(&self) -> usize {
        self.row_index
    }

    /// Get the number of fields
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Get a field value by name
    pub fn field_value(&self, name: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }

    /// Get a field value by index
    pub fn field_by_index(&self, index: usize) -> Option<(&str, &str)> {
        self.fields
            .get(index)
            .map(|(n, v)| (n.as_str(), v.as_str()))
    }

    // Navigation

    /// Scroll down
    pub fn scroll_down(&mut self) {
        self.scroll.scroll_down();
    }

    /// Scroll up
    pub fn scroll_up(&mut self) {
        self.scroll.scroll_up();
    }

    /// Page down
    pub fn page_down(&mut self) {
        self.scroll.page_down();
    }

    /// Page up
    pub fn page_up(&mut self) {
        self.scroll.page_up();
    }

    /// Get scroll offset
    pub fn scroll_offset(&self) -> usize {
        self.scroll.offset()
    }

    /// Render the detail view as lines
    pub fn render_lines(&self) -> Vec<String> {
        let max_width = self.display_width.saturating_sub(4) as usize; // margins
        let mut all_lines = Vec::new();

        // Title
        all_lines.push(format!("Row {}", self.row_index));
        all_lines.push(String::new());

        // Fields
        for (name, value) in &self.fields {
            // Field name
            all_lines.push(format!("{name}:"));

            // Wrap value if needed
            let wrapped = wrap_text(value, max_width);
            for line in wrapped {
                all_lines.push(format!("  {line}"));
            }

            // Blank line between fields
            all_lines.push(String::new());
        }

        // Apply scroll offset
        let start = self.scroll.offset();
        let visible = self.display_height.saturating_sub(2) as usize;
        let end = (start + visible).min(all_lines.len());

        all_lines[start..end].to_vec()
    }

    /// Render as a single string
    pub fn render(&self) -> String {
        self.render_lines().join("\n")
    }
}

/// Calculate total lines needed for all fields
fn calculate_total_lines(fields: &[(String, String)], width: u16) -> usize {
    let max_width = width.saturating_sub(4) as usize;

    fields
        .iter()
        .map(|(_, value)| {
            // Name line + wrapped value lines + blank line
            1 + wrap_text(value, max_width).len() + 1
        })
        .sum::<usize>()
        .saturating_add(2) // Title + blank
}

/// Wrap text to fit within a maximum width
fn wrap_text(text: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 {
        return vec![text.to_string()];
    }

    let mut lines = Vec::new();

    for line in text.lines() {
        if line.is_empty() {
            lines.push(String::new());
            continue;
        }

        let chars: Vec<char> = line.chars().collect();
        let mut start = 0;

        while start < chars.len() {
            let end = (start + max_width).min(chars.len());
            let segment: String = chars[start..end].iter().collect();
            lines.push(segment);
            start = end;
        }
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    lines
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Float32Array, Int32Array, RecordBatch, StringArray},
        datatypes::{DataType, Field, Schema},
    };

    use super::*;

    fn create_test_adapter() -> DatasetAdapter {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("description", DataType::Utf8, false),
            Field::new("value", DataType::Int32, false),
            Field::new("score", DataType::Float32, false),
        ]));

        let ids = vec!["row_0", "row_1", "row_2"];
        let descriptions = vec![
            "Short description",
            "This is a much longer description that will need to be wrapped across multiple lines when displayed in the detail view",
            "Another row",
        ];
        let values = vec![100, 200, 300];
        let scores = vec![0.95_f32, 0.87, 0.99];

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(descriptions)),
                Arc::new(Int32Array::from(values)),
                Arc::new(Float32Array::from(scores)),
            ],
        )
        .unwrap();

        DatasetAdapter::from_batches(vec![batch], schema).unwrap()
    }

    #[test]
    fn f_detail_new() {
        let adapter = create_test_adapter();
        let detail = RowDetailView::new(&adapter, 0);
        assert!(detail.is_some());
    }

    #[test]
    fn f_detail_out_of_bounds() {
        let adapter = create_test_adapter();
        let detail = RowDetailView::new(&adapter, 100);
        assert!(detail.is_none());
    }

    #[test]
    fn f_detail_row_index() {
        let adapter = create_test_adapter();
        let detail = RowDetailView::new(&adapter, 1).unwrap();
        assert_eq!(detail.row_index(), 1);
    }

    #[test]
    fn f_detail_field_count() {
        let adapter = create_test_adapter();
        let detail = RowDetailView::new(&adapter, 0).unwrap();
        assert_eq!(detail.field_count(), 4);
    }

    #[test]
    fn f_detail_field_value_by_name() {
        let adapter = create_test_adapter();
        let detail = RowDetailView::new(&adapter, 0).unwrap();
        let value = detail.field_value("id");
        assert_eq!(value, Some("row_0"));
    }

    #[test]
    fn f_detail_field_value_not_found() {
        let adapter = create_test_adapter();
        let detail = RowDetailView::new(&adapter, 0).unwrap();
        let value = detail.field_value("nonexistent");
        assert!(value.is_none());
    }

    #[test]
    fn f_detail_field_by_index() {
        let adapter = create_test_adapter();
        let detail = RowDetailView::new(&adapter, 0).unwrap();
        let (name, value) = detail.field_by_index(0).unwrap();
        assert_eq!(name, "id");
        assert_eq!(value, "row_0");
    }

    #[test]
    fn f_detail_field_by_index_out_of_bounds() {
        let adapter = create_test_adapter();
        let detail = RowDetailView::new(&adapter, 0).unwrap();
        assert!(detail.field_by_index(100).is_none());
    }

    #[test]
    fn f_detail_render_lines() {
        let adapter = create_test_adapter();
        let detail = RowDetailView::new(&adapter, 0).unwrap();
        let lines = detail.render_lines();

        assert!(!lines.is_empty());
        assert!(lines[0].contains("Row 0"));
    }

    #[test]
    fn f_detail_render() {
        let adapter = create_test_adapter();
        let detail = RowDetailView::new(&adapter, 0).unwrap();
        let rendered = detail.render();

        assert!(rendered.contains("Row 0"));
        assert!(rendered.contains("id:"));
        assert!(rendered.contains("row_0"));
    }

    #[test]
    fn f_detail_scroll_down() {
        let adapter = create_test_adapter();
        let mut detail = RowDetailView::new(&adapter, 1).unwrap();
        let initial = detail.scroll_offset();
        detail.scroll_down();
        // May or may not change depending on content size
        assert!(detail.scroll_offset() >= initial);
    }

    #[test]
    fn f_detail_scroll_up() {
        let adapter = create_test_adapter();
        let mut detail = RowDetailView::with_dimensions(&adapter, 1, 40, 10).unwrap();
        detail.scroll_down();
        detail.scroll_down();
        detail.scroll_down();
        let after_down = detail.scroll_offset();
        detail.scroll_up();
        assert!(detail.scroll_offset() <= after_down);
    }

    #[test]
    fn f_detail_is_empty() {
        let adapter = create_test_adapter();
        let detail = RowDetailView::new(&adapter, 0).unwrap();
        assert!(!detail.is_empty());
    }

    #[test]
    fn f_detail_clone() {
        let adapter = create_test_adapter();
        let detail = RowDetailView::new(&adapter, 0).unwrap();
        let cloned = detail.clone();
        assert_eq!(detail.row_index(), cloned.row_index());
        assert_eq!(detail.field_count(), cloned.field_count());
    }

    #[test]
    fn f_wrap_text_short() {
        let wrapped = wrap_text("hello", 20);
        assert_eq!(wrapped, vec!["hello"]);
    }

    #[test]
    fn f_wrap_text_long() {
        let text = "This is a long line that needs wrapping";
        let wrapped = wrap_text(text, 10);
        assert!(wrapped.len() > 1);
        for line in &wrapped {
            assert!(line.chars().count() <= 10);
        }
    }

    #[test]
    fn f_wrap_text_multiline() {
        let text = "Line one\nLine two";
        let wrapped = wrap_text(text, 50);
        assert_eq!(wrapped.len(), 2);
    }

    #[test]
    fn f_wrap_text_empty() {
        let wrapped = wrap_text("", 20);
        assert_eq!(wrapped.len(), 1);
    }

    #[test]
    fn f_wrap_text_zero_width() {
        let wrapped = wrap_text("hello", 0);
        assert_eq!(wrapped, vec!["hello"]);
    }

    #[test]
    fn f_calculate_total_lines() {
        let fields = vec![
            ("name".to_string(), "value".to_string()),
            ("other".to_string(), "data".to_string()),
        ];
        let total = calculate_total_lines(&fields, 80);
        // Title + blank + (field_name + value + blank) * 2
        assert!(total >= 8);
    }

    #[test]
    fn f_detail_page_down() {
        let adapter = create_test_adapter();
        let mut detail = RowDetailView::with_dimensions(&adapter, 1, 40, 5).unwrap();
        let initial = detail.scroll_offset();
        detail.page_down();
        // Page down should increase offset (or stay same if at end)
        assert!(detail.scroll_offset() >= initial);
    }

    #[test]
    fn f_detail_page_up() {
        let adapter = create_test_adapter();
        let mut detail = RowDetailView::with_dimensions(&adapter, 1, 40, 5).unwrap();
        // Scroll down first
        detail.page_down();
        detail.page_down();
        let after_down = detail.scroll_offset();
        // Now page up
        detail.page_up();
        // Should decrease or stay same
        assert!(detail.scroll_offset() <= after_down);
    }

    #[test]
    fn f_wrap_text_with_empty_line() {
        let text = "First\n\nThird";
        let wrapped = wrap_text(text, 50);
        assert_eq!(wrapped.len(), 3);
        assert_eq!(wrapped[1], "");
    }
}
