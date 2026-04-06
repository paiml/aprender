//! Dataset viewer widget for TUI display
//!
//! Provides a scrollable table view of Arrow datasets.

use super::{adapter::DatasetAdapter, format::truncate_string, scroll::ScrollState};

/// A scrollable table view for displaying Arrow datasets
///
/// The viewer provides:
/// - Scrollable rows with keyboard navigation
/// - Column headers with field names
/// - Selected row highlighting
/// - Automatic column width calculation
/// - Truncation with ellipsis for long values
///
/// # Example
///
/// ```ignore
/// let adapter = DatasetAdapter::from_dataset(&dataset)?;
/// let viewer = DatasetViewer::new(adapter);
///
/// // Handle keyboard input
/// viewer.scroll_down();
/// viewer.select_row(5);
///
/// // Get visible data for rendering
/// let headers = viewer.headers();
/// let rows = viewer.visible_rows();
/// ```
#[derive(Debug, Clone)]
pub struct DatasetViewer {
    /// The dataset adapter
    adapter: DatasetAdapter,
    /// Scroll state
    scroll: ScrollState,
    /// Calculated column widths
    column_widths: Vec<u16>,
    /// Total display width
    display_width: u16,
    /// Number of visible rows (excluding header)
    visible_rows: u16,
}

impl DatasetViewer {
    /// Create a new viewer with default dimensions
    ///
    /// # Arguments
    /// * `adapter` - The dataset adapter to display
    pub fn new(adapter: DatasetAdapter) -> Self {
        Self::with_dimensions(adapter, 80, 24)
    }

    /// Create a new viewer with specific dimensions
    ///
    /// # Arguments
    /// * `adapter` - The dataset adapter to display
    /// * `width` - Display width in characters
    /// * `height` - Display height in rows (including header)
    pub fn with_dimensions(adapter: DatasetAdapter, width: u16, height: u16) -> Self {
        let visible_rows = height.saturating_sub(1); // -1 for header
        let column_widths = adapter.calculate_column_widths(width, 20);
        let scroll = ScrollState::new(adapter.row_count(), visible_rows as usize);

        Self {
            adapter,
            scroll,
            column_widths,
            display_width: width,
            visible_rows,
        }
    }

    /// Update display dimensions
    ///
    /// Recalculates column widths and scroll state.
    pub fn set_dimensions(&mut self, width: u16, height: u16) {
        self.display_width = width;
        self.visible_rows = height.saturating_sub(1);
        self.column_widths = self.adapter.calculate_column_widths(width, 20);
        self.scroll.set_visible_rows(self.visible_rows as usize);
    }

    /// Get the current scroll offset
    #[inline]
    pub fn scroll_offset(&self) -> usize {
        self.scroll.offset()
    }

    /// Set the scroll offset
    pub fn set_scroll_offset(&mut self, offset: usize) {
        self.scroll.set_offset(offset);
    }

    /// Get total row count
    #[inline]
    pub fn row_count(&self) -> usize {
        self.adapter.row_count()
    }

    /// Get visible row count
    #[inline]
    pub fn visible_row_count(&self) -> u16 {
        self.visible_rows
    }

    /// Get the currently selected row
    #[inline]
    pub fn selected_row(&self) -> Option<usize> {
        self.scroll.selected()
    }

    /// Select a specific row
    pub fn select_row(&mut self, row: usize) {
        self.scroll.set_selected(Some(row));
    }

    /// Clear selection
    pub fn clear_selection(&mut self) {
        self.scroll.set_selected(None);
    }

    /// Check if the dataset is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.adapter.is_empty()
    }

    /// Get the adapter reference
    #[inline]
    pub fn adapter(&self) -> &DatasetAdapter {
        &self.adapter
    }

    /// Get column widths
    #[inline]
    pub fn column_widths(&self) -> &[u16] {
        &self.column_widths
    }

    // Navigation methods

    /// Scroll down by one row
    pub fn scroll_down(&mut self) {
        self.scroll.scroll_down();
    }

    /// Scroll up by one row
    pub fn scroll_up(&mut self) {
        self.scroll.scroll_up();
    }

    /// Scroll down by one page
    pub fn page_down(&mut self) {
        self.scroll.page_down();
    }

    /// Scroll up by one page
    pub fn page_up(&mut self) {
        self.scroll.page_up();
    }

    /// Jump to first row
    pub fn home(&mut self) {
        self.scroll.home();
    }

    /// Jump to last page
    pub fn end(&mut self) {
        self.scroll.end();
    }

    /// Select next row
    pub fn select_next(&mut self) {
        self.scroll.select_next();
    }

    /// Select previous row
    pub fn select_prev(&mut self) {
        self.scroll.select_prev();
    }

    // Rendering helpers

    /// Get column headers
    pub fn headers(&self) -> Vec<String> {
        self.adapter
            .field_names()
            .into_iter()
            .enumerate()
            .map(|(i, name)| {
                let width = self.column_widths.get(i).copied().unwrap_or(10) as usize;
                truncate_string(name, width)
            })
            .collect()
    }

    /// Get visible rows as formatted strings
    ///
    /// Returns a vector of rows, where each row is a vector of cell strings.
    pub fn visible_rows_data(&self) -> Vec<Vec<String>> {
        let start = self.scroll.offset();
        let end = (start + self.visible_rows as usize).min(self.adapter.row_count());

        (start..end)
            .map(|row_idx| self.format_row(row_idx))
            .collect()
    }

    /// Format a single row
    fn format_row(&self, row_idx: usize) -> Vec<String> {
        (0..self.adapter.column_count())
            .map(|col_idx| {
                let width = self.column_widths.get(col_idx).copied().unwrap_or(10) as usize;
                match self.adapter.get_cell(row_idx, col_idx) {
                    Ok(Some(value)) => truncate_string(&value, width),
                    Ok(None) => String::new(),
                    Err(_) => "<error>".to_string(),
                }
            })
            .collect()
    }

    /// Check if a row is currently selected
    pub fn is_row_selected(&self, global_row: usize) -> bool {
        self.scroll.selected() == Some(global_row)
    }

    /// Check if scrollbar should be shown
    pub fn needs_scrollbar(&self) -> bool {
        self.scroll.needs_scrollbar()
    }

    /// Get scrollbar position (0.0 to 1.0)
    pub fn scrollbar_position(&self) -> f32 {
        self.scroll.scrollbar_position()
    }

    /// Get scrollbar size (0.0 to 1.0)
    pub fn scrollbar_size(&self) -> f32 {
        self.scroll.scrollbar_size()
    }

    /// Render header line as a string
    pub fn render_header_line(&self) -> String {
        let headers = self.headers();
        headers.join(" ")
    }

    /// Render a data row as a string
    pub fn render_row_line(&self, viewport_row: usize) -> Option<String> {
        let global_row = self.scroll.to_global_row(viewport_row);
        if global_row >= self.adapter.row_count() {
            return None;
        }

        let cells = self.format_row(global_row);
        Some(cells.join(" "))
    }

    /// Get the data row index for a viewport row
    pub fn viewport_to_data_row(&self, viewport_row: usize) -> usize {
        self.scroll.to_global_row(viewport_row)
    }

    // Search methods

    /// Search for a substring and select the first matching row
    ///
    /// Returns the row index if found, None otherwise.
    /// This is a linear scan suitable for datasets <100k rows (F101).
    pub fn search(&mut self, query: &str) -> Option<usize> {
        let result = self.adapter.search(query);
        if let Some(row) = result {
            self.select_row(row);
            self.scroll.ensure_visible(row);
        }
        result
    }

    /// Continue search from current position
    ///
    /// Wraps around to beginning if no match found after current row.
    pub fn search_next(&mut self, query: &str) -> Option<usize> {
        let start = self.scroll.selected().map(|r| r + 1).unwrap_or(0);
        let result = self.adapter.search_from(query, start);
        if let Some(row) = result {
            self.select_row(row);
            self.scroll.ensure_visible(row);
        }
        result
    }

    /// Render complete output as lines
    ///
    /// Returns header followed by visible data rows.
    pub fn render_lines(&self) -> Vec<String> {
        let mut lines = Vec::with_capacity(self.visible_rows as usize + 1);

        // Header
        lines.push(self.render_header_line());

        // Data rows
        for vrow in 0..self.visible_rows as usize {
            if let Some(line) = self.render_row_line(vrow) {
                lines.push(line);
            }
        }

        lines
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Float32Array, Int32Array, RecordBatch, StringArray},
        datatypes::{DataType, Field, Schema},
    };

    use super::*;

    fn create_test_adapter(rows: usize) -> DatasetAdapter {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("value", DataType::Int32, false),
            Field::new("score", DataType::Float32, false),
        ]));

        let ids: Vec<String> = (0..rows).map(|i| format!("id_{i}")).collect();
        let values: Vec<i32> = (0..rows).map(|i| i as i32 * 10).collect();
        let scores: Vec<f32> = (0..rows).map(|i| i as f32 * 0.1).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(Int32Array::from(values)),
                Arc::new(Float32Array::from(scores)),
            ],
        )
        .unwrap();

        DatasetAdapter::from_batches(vec![batch], schema).unwrap()
    }

    fn create_test_viewer() -> DatasetViewer {
        let adapter = create_test_adapter(80);
        DatasetViewer::with_dimensions(adapter, 80, 24)
    }

    #[test]
    fn f026_viewer_new() {
        let viewer = create_test_viewer();
        assert_eq!(viewer.row_count(), 80);
        assert_eq!(viewer.scroll_offset(), 0);
    }

    #[test]
    fn f027_viewer_scroll_down() {
        let mut viewer = create_test_viewer();
        viewer.scroll_down();
        assert_eq!(viewer.scroll_offset(), 1);
    }

    #[test]
    fn f028_viewer_scroll_up() {
        let mut viewer = create_test_viewer();
        viewer.set_scroll_offset(10);
        viewer.scroll_up();
        assert_eq!(viewer.scroll_offset(), 9);
    }

    #[test]
    fn f029_viewer_scroll_bounds_top() {
        let mut viewer = create_test_viewer();
        viewer.scroll_up();
        assert_eq!(viewer.scroll_offset(), 0);
    }

    #[test]
    fn f030_viewer_page_down() {
        let mut viewer = create_test_viewer();
        let visible = viewer.visible_row_count() as usize;
        viewer.page_down();
        assert!(viewer.scroll_offset() >= visible / 2);
    }

    #[test]
    fn f031_viewer_page_up() {
        let mut viewer = create_test_viewer();
        viewer.set_scroll_offset(50);
        viewer.page_up();
        assert!(viewer.scroll_offset() < 50);
    }

    #[test]
    fn f032_viewer_home() {
        let mut viewer = create_test_viewer();
        viewer.set_scroll_offset(50);
        viewer.home();
        assert_eq!(viewer.scroll_offset(), 0);
    }

    #[test]
    fn f033_viewer_end() {
        let mut viewer = create_test_viewer();
        viewer.end();
        // Should be at position where last row is visible
        let max_offset = viewer.row_count() - viewer.visible_row_count() as usize;
        assert_eq!(viewer.scroll_offset(), max_offset);
    }

    #[test]
    fn f034_viewer_select_row() {
        let mut viewer = create_test_viewer();
        viewer.select_row(5);
        assert_eq!(viewer.selected_row(), Some(5));
    }

    #[test]
    fn f035_viewer_select_next() {
        let mut viewer = create_test_viewer();
        viewer.select_row(0);
        viewer.select_next();
        assert_eq!(viewer.selected_row(), Some(1));
    }

    #[test]
    fn f036_viewer_select_prev() {
        let mut viewer = create_test_viewer();
        viewer.select_row(5);
        viewer.select_prev();
        assert_eq!(viewer.selected_row(), Some(4));
    }

    #[test]
    fn f037_viewer_clear_selection() {
        let mut viewer = create_test_viewer();
        viewer.select_row(5);
        viewer.clear_selection();
        assert_eq!(viewer.selected_row(), None);
    }

    #[test]
    fn f038_viewer_headers() {
        let viewer = create_test_viewer();
        let headers = viewer.headers();
        assert_eq!(headers.len(), 3);
        assert!(headers[0].contains("id"));
    }

    #[test]
    fn f039_viewer_visible_rows() {
        let viewer = create_test_viewer();
        let rows = viewer.visible_rows_data();
        assert!(rows.len() <= viewer.visible_row_count() as usize);
    }

    #[test]
    fn f040_viewer_needs_scrollbar() {
        let viewer = create_test_viewer();
        assert!(viewer.needs_scrollbar());
    }

    #[test]
    fn f041_viewer_no_scrollbar_small() {
        let adapter = create_test_adapter(5);
        let viewer = DatasetViewer::with_dimensions(adapter, 80, 24);
        assert!(!viewer.needs_scrollbar());
    }

    #[test]
    fn f042_viewer_scrollbar_position() {
        let mut viewer = create_test_viewer();
        viewer.set_scroll_offset(40);
        let pos = viewer.scrollbar_position();
        assert!(pos > 0.0 && pos < 1.0);
    }

    #[test]
    fn f043_viewer_render_header() {
        let viewer = create_test_viewer();
        let header = viewer.render_header_line();
        assert!(!header.is_empty());
        assert!(header.contains("id"));
    }

    #[test]
    fn f044_viewer_render_row() {
        let viewer = create_test_viewer();
        let row = viewer.render_row_line(0);
        assert!(row.is_some());
        assert!(row.unwrap().contains("id_0"));
    }

    #[test]
    fn f045_viewer_render_row_out_of_bounds() {
        let viewer = create_test_viewer();
        let row = viewer.render_row_line(1000);
        assert!(row.is_none());
    }

    #[test]
    fn f046_viewer_render_lines() {
        let viewer = create_test_viewer();
        let lines = viewer.render_lines();
        assert!(!lines.is_empty());
        // First line is header
        assert!(lines[0].contains("id"));
    }

    #[test]
    fn f047_viewer_column_widths() {
        let viewer = create_test_viewer();
        let widths = viewer.column_widths();
        assert_eq!(widths.len(), 3);
        for w in widths {
            assert!(*w >= 3);
        }
    }

    #[test]
    fn f048_viewer_is_row_selected() {
        let mut viewer = create_test_viewer();
        viewer.select_row(5);
        assert!(viewer.is_row_selected(5));
        assert!(!viewer.is_row_selected(4));
    }

    #[test]
    fn f049_viewer_set_dimensions() {
        let mut viewer = create_test_viewer();
        viewer.set_dimensions(40, 10);
        assert_eq!(viewer.visible_row_count(), 9);
    }

    #[test]
    fn f050_viewer_empty() {
        let adapter = DatasetAdapter::empty();
        let viewer = DatasetViewer::new(adapter);
        assert!(viewer.is_empty());
        assert_eq!(viewer.row_count(), 0);
    }

    #[test]
    fn f_viewer_viewport_to_data_row() {
        let mut viewer = create_test_viewer();
        viewer.set_scroll_offset(10);
        assert_eq!(viewer.viewport_to_data_row(5), 15);
    }

    #[test]
    fn f_viewer_is_clone() {
        let viewer = create_test_viewer();
        let cloned = viewer.clone();
        assert_eq!(viewer.row_count(), cloned.row_count());
    }

    #[test]
    fn f_viewer_adapter_access() {
        let viewer = create_test_viewer();
        let adapter = viewer.adapter();
        assert_eq!(adapter.column_count(), 3);
    }

    #[test]
    fn f_viewer_scrollbar_size() {
        let viewer = create_test_viewer();
        let size = viewer.scrollbar_size();
        // Should be between 0 and 1
        assert!(size > 0.0 && size <= 1.0);
    }

    #[test]
    fn f_viewer_scrollbar_size_small_dataset() {
        let adapter = create_test_adapter(5);
        let viewer = DatasetViewer::with_dimensions(adapter, 80, 24);
        let size = viewer.scrollbar_size();
        // When all content fits, size should be 1.0
        assert!((size - 1.0).abs() < 0.01);
    }

    #[test]
    fn f_viewer_format_row_with_null() {
        // Create a dataset with nullable column containing null
        use arrow::array::NullArray;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("nullable_col", DataType::Null, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a", "b"])),
                Arc::new(NullArray::new(2)),
            ],
        )
        .unwrap();

        let adapter = DatasetAdapter::from_batches(vec![batch], schema).unwrap();
        let viewer = DatasetViewer::new(adapter);
        let rows = viewer.visible_rows_data();

        // FALSIFIABLE: Assert actual behavior, not just existence
        assert_eq!(rows.len(), 2, "FALSIFIED: Should have 2 rows");
        assert_eq!(rows[0].len(), 2, "FALSIFIED: Should have 2 columns");
        assert_eq!(rows[0][0], "a", "FALSIFIED: First cell should be 'a'");
        // Null column should render as empty string or "NULL"
        assert!(
            rows[0][1].is_empty() || rows[0][1] == "null" || rows[0][1] == "NULL",
            "FALSIFIED: Null should render as empty or 'null'/'NULL', got: '{}'",
            rows[0][1]
        );
    }

    // === SEARCH TESTS (F101-F103) ===

    #[test]
    fn f_viewer_search_finds_match() {
        let mut viewer = create_test_viewer();
        let result = viewer.search("id_5");
        assert_eq!(
            result,
            Some(5),
            "FALSIFIED: Search should find 'id_5' at row 5"
        );
        assert_eq!(
            viewer.selected_row(),
            Some(5),
            "FALSIFIED: Search should select found row"
        );
    }

    #[test]
    fn f_viewer_search_no_match() {
        let mut viewer = create_test_viewer();
        let result = viewer.search("nonexistent_xyz");
        assert_eq!(result, None, "FALSIFIED: Search should return None");
        assert_eq!(
            viewer.selected_row(),
            None,
            "FALSIFIED: No selection should change"
        );
    }

    #[test]
    fn f_viewer_search_case_insensitive() {
        let mut viewer = create_test_viewer();
        let result = viewer.search("ID_3");
        assert_eq!(
            result,
            Some(3),
            "FALSIFIED: Search should be case insensitive"
        );
    }

    #[test]
    fn f_viewer_search_next_continues() {
        let mut viewer = create_test_viewer();
        // First search
        viewer.search("id_");
        let first = viewer.selected_row();
        // Search next should find a different row
        viewer.search_next("id_");
        let second = viewer.selected_row();
        assert_ne!(first, second, "FALSIFIED: search_next should continue");
    }

    #[test]
    fn f_viewer_search_next_wraps() {
        let mut viewer = create_test_viewer();
        // Select last row
        viewer.select_row(9);
        // Search next should wrap to beginning
        let result = viewer.search_next("id_0");
        assert_eq!(result, Some(0), "FALSIFIED: search_next should wrap");
    }

    #[test]
    fn f_viewer_search_empty_query() {
        let mut viewer = create_test_viewer();
        let result = viewer.search("");
        assert_eq!(result, None, "FALSIFIED: Empty query should return None");
    }
}
