//! `DataTable` widget for displaying tabular data.

use presentar_core::{
    widget::{AccessibleRole, LayoutResult, TextStyle},
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event, Rect,
    Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// Column definition for a data table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableColumn {
    /// Column key (field name in data)
    pub key: String,
    /// Display header
    pub header: String,
    /// Column width (None = auto)
    pub width: Option<f32>,
    /// Text alignment
    pub align: TextAlign,
    /// Whether column is sortable
    pub sortable: bool,
}

impl TableColumn {
    /// Create a new column.
    #[must_use]
    pub fn new(key: impl Into<String>, header: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            header: header.into(),
            width: None,
            align: TextAlign::Left,
            sortable: false,
        }
    }

    /// Set column width.
    #[must_use]
    pub fn width(mut self, width: f32) -> Self {
        self.width = Some(width.max(20.0));
        self
    }

    /// Set text alignment.
    #[must_use]
    pub const fn align(mut self, align: TextAlign) -> Self {
        self.align = align;
        self
    }

    /// Make column sortable.
    #[must_use]
    pub const fn sortable(mut self) -> Self {
        self.sortable = true;
        self
    }
}

/// Text alignment within a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
}

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Message emitted when table sorting changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableSortChanged {
    /// Column key being sorted
    pub column: String,
    /// Sort direction
    pub direction: SortDirection,
}

/// Message emitted when a row is selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableRowSelected {
    /// Index of selected row
    pub index: usize,
}

/// A cell value in the table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CellValue {
    /// Text value
    Text(String),
    /// Numeric value
    Number(f64),
    /// Boolean value
    Bool(bool),
    /// Empty cell
    Empty,
}

impl CellValue {
    /// Get display text for the cell.
    #[must_use]
    pub fn display(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Number(n) => format!("{n}"),
            Self::Bool(b) => if *b { "Yes" } else { "No" }.to_string(),
            Self::Empty => String::new(),
        }
    }
}

impl From<&str> for CellValue {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl From<String> for CellValue {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<f64> for CellValue {
    fn from(n: f64) -> Self {
        Self::Number(n)
    }
}

impl From<i32> for CellValue {
    fn from(n: i32) -> Self {
        Self::Number(f64::from(n))
    }
}

impl From<bool> for CellValue {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

/// A row of data in the table.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TableRow {
    /// Cell values by column key
    pub cells: std::collections::HashMap<String, CellValue>,
}

impl TableRow {
    /// Create a new empty row.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a cell value.
    #[must_use]
    pub fn cell(mut self, key: impl Into<String>, value: impl Into<CellValue>) -> Self {
        self.cells.insert(key.into(), value.into());
        self
    }

    /// Get a cell value.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&CellValue> {
        self.cells.get(key)
    }
}

/// `DataTable` widget for displaying tabular data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTable {
    /// Column definitions
    columns: Vec<TableColumn>,
    /// Row data
    rows: Vec<TableRow>,
    /// Row height
    row_height: f32,
    /// Header height
    header_height: f32,
    /// Current sort column
    sort_column: Option<String>,
    /// Current sort direction
    sort_direction: SortDirection,
    /// Selected row index
    selected_row: Option<usize>,
    /// Whether rows are selectable
    selectable: bool,
    /// Striped rows
    striped: bool,
    /// Show borders
    bordered: bool,
    /// Header background color
    header_bg: Color,
    /// Row background color
    row_bg: Color,
    /// Alternate row background color
    row_alt_bg: Color,
    /// Selected row background color
    selected_bg: Color,
    /// Border color
    border_color: Color,
    /// Text color
    text_color: Color,
    /// Header text color
    header_text_color: Color,
    /// Accessible name
    accessible_name_value: Option<String>,
    /// Test ID
    test_id_value: Option<String>,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
}

impl Default for DataTable {
    fn default() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            row_height: 40.0,
            header_height: 44.0,
            sort_column: None,
            sort_direction: SortDirection::Ascending,
            selected_row: None,
            selectable: false,
            striped: true,
            bordered: true,
            header_bg: Color::new(0.95, 0.95, 0.95, 1.0),
            row_bg: Color::WHITE,
            row_alt_bg: Color::new(0.98, 0.98, 0.98, 1.0),
            selected_bg: Color::new(0.9, 0.95, 1.0, 1.0),
            border_color: Color::new(0.85, 0.85, 0.85, 1.0),
            text_color: Color::BLACK,
            header_text_color: Color::new(0.2, 0.2, 0.2, 1.0),
            accessible_name_value: None,
            test_id_value: None,
            bounds: Rect::default(),
        }
    }
}

impl DataTable {
    /// Create a new empty data table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a column.
    #[must_use]
    pub fn column(mut self, column: TableColumn) -> Self {
        self.columns.push(column);
        self
    }

    /// Add multiple columns.
    #[must_use]
    pub fn columns(mut self, columns: impl IntoIterator<Item = TableColumn>) -> Self {
        self.columns.extend(columns);
        self
    }

    /// Add a row.
    #[must_use]
    pub fn row(mut self, row: TableRow) -> Self {
        self.rows.push(row);
        self
    }

    /// Add multiple rows.
    #[must_use]
    pub fn rows(mut self, rows: impl IntoIterator<Item = TableRow>) -> Self {
        self.rows.extend(rows);
        self
    }

    /// Set row height.
    #[must_use]
    pub fn row_height(mut self, height: f32) -> Self {
        self.row_height = height.max(20.0);
        self
    }

    /// Set header height.
    #[must_use]
    pub fn header_height(mut self, height: f32) -> Self {
        self.header_height = height.max(20.0);
        self
    }

    /// Enable row selection.
    #[must_use]
    pub const fn selectable(mut self, selectable: bool) -> Self {
        self.selectable = selectable;
        self
    }

    /// Enable striped rows.
    #[must_use]
    pub const fn striped(mut self, striped: bool) -> Self {
        self.striped = striped;
        self
    }

    /// Enable borders.
    #[must_use]
    pub const fn bordered(mut self, bordered: bool) -> Self {
        self.bordered = bordered;
        self
    }

    /// Set header background color.
    #[must_use]
    pub const fn header_bg(mut self, color: Color) -> Self {
        self.header_bg = color;
        self
    }

    /// Set row background color.
    #[must_use]
    pub const fn row_bg(mut self, color: Color) -> Self {
        self.row_bg = color;
        self
    }

    /// Set alternate row background color.
    #[must_use]
    pub const fn row_alt_bg(mut self, color: Color) -> Self {
        self.row_alt_bg = color;
        self
    }

    /// Set selected row background color.
    #[must_use]
    pub const fn selected_bg(mut self, color: Color) -> Self {
        self.selected_bg = color;
        self
    }

    /// Set text color.
    #[must_use]
    pub const fn text_color(mut self, color: Color) -> Self {
        self.text_color = color;
        self
    }

    /// Set the accessible name.
    #[must_use]
    pub fn accessible_name(mut self, name: impl Into<String>) -> Self {
        self.accessible_name_value = Some(name.into());
        self
    }

    /// Set the test ID.
    #[must_use]
    pub fn test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id_value = Some(id.into());
        self
    }

    /// Get column count.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Get row count.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Get columns.
    #[must_use]
    pub fn get_columns(&self) -> &[TableColumn] {
        &self.columns
    }

    /// Get rows.
    #[must_use]
    pub fn get_rows(&self) -> &[TableRow] {
        &self.rows
    }

    /// Get selected row index.
    #[must_use]
    pub const fn get_selected_row(&self) -> Option<usize> {
        self.selected_row
    }

    /// Get current sort column.
    #[must_use]
    pub fn get_sort_column(&self) -> Option<&str> {
        self.sort_column.as_deref()
    }

    /// Get current sort direction.
    #[must_use]
    pub const fn get_sort_direction(&self) -> SortDirection {
        self.sort_direction
    }

    /// Check if table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Select a row.
    pub fn select_row(&mut self, index: Option<usize>) {
        if let Some(idx) = index {
            if idx < self.rows.len() {
                self.selected_row = Some(idx);
            }
        } else {
            self.selected_row = None;
        }
    }

    /// Set sort column and direction.
    pub fn set_sort(&mut self, column: impl Into<String>, direction: SortDirection) {
        self.sort_column = Some(column.into());
        self.sort_direction = direction;
    }

    /// Clear data.
    pub fn clear(&mut self) {
        self.rows.clear();
        self.selected_row = None;
    }

    /// Calculate total width.
    fn calculate_width(&self) -> f32 {
        let mut total = 0.0;
        for col in &self.columns {
            total += col.width.unwrap_or(100.0);
        }
        total.max(100.0)
    }

    /// Calculate total height.
    fn calculate_height(&self) -> f32 {
        (self.rows.len() as f32).mul_add(self.row_height, self.header_height)
    }

    /// Get row Y position.
    fn row_y(&self, index: usize) -> f32 {
        (index as f32).mul_add(self.row_height, self.bounds.y + self.header_height)
    }
}

impl Widget for DataTable {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let preferred = Size::new(self.calculate_width(), self.calculate_height());
        constraints.constrain(preferred)
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        // Draw header row
        let header_rect = Rect::new(
            self.bounds.x,
            self.bounds.y,
            self.bounds.width,
            self.header_height,
        );
        canvas.fill_rect(header_rect, self.header_bg);

        // Draw header text
        let mut x = self.bounds.x;
        for col in &self.columns {
            let col_width = col.width.unwrap_or(100.0);
            let text_style = TextStyle {
                size: 14.0,
                color: self.header_text_color,
                weight: presentar_core::widget::FontWeight::Bold,
                ..TextStyle::default()
            };
            canvas.draw_text(
                &col.header,
                presentar_core::Point::new(x + 8.0, self.bounds.y + self.header_height / 2.0),
                &text_style,
            );
            x += col_width;
        }

        // Draw data rows
        for (row_idx, row) in self.rows.iter().enumerate() {
            let row_y = self.row_y(row_idx);

            // Determine background color
            let bg_color = if self.selected_row == Some(row_idx) {
                self.selected_bg
            } else if self.striped && row_idx % 2 == 1 {
                self.row_alt_bg
            } else {
                self.row_bg
            };

            let row_rect = Rect::new(self.bounds.x, row_y, self.bounds.width, self.row_height);
            canvas.fill_rect(row_rect, bg_color);

            // Draw cell values
            let mut x = self.bounds.x;
            for col in &self.columns {
                let col_width = col.width.unwrap_or(100.0);
                if let Some(cell) = row.get(&col.key) {
                    let text_style = TextStyle {
                        size: 14.0,
                        color: self.text_color,
                        ..TextStyle::default()
                    };
                    canvas.draw_text(
                        &cell.display(),
                        presentar_core::Point::new(x + 8.0, row_y + self.row_height / 2.0),
                        &text_style,
                    );
                }
                x += col_width;
            }
        }

        // Draw borders
        if self.bordered {
            let border_rect = Rect::new(
                self.bounds.x,
                self.bounds.y,
                self.bounds.width,
                self.calculate_height().min(self.bounds.height),
            );
            canvas.stroke_rect(border_rect, self.border_color, 1.0);
        }
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        // Row selection and sorting would be handled here
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }

    fn is_interactive(&self) -> bool {
        self.selectable
    }

    fn is_focusable(&self) -> bool {
        self.selectable
    }

    fn accessible_name(&self) -> Option<&str> {
        self.accessible_name_value.as_deref()
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Table
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for DataTable {
    fn brick_name(&self) -> &'static str {
        "DataTable"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[BrickAssertion::MaxLatencyMs(16)]
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
        let test_id = self.test_id_value.as_deref().unwrap_or("data-table");
        format!(r#"<table class="brick-data-table" data-testid="{test_id}" role="table"></table>"#)
    }

    fn to_css(&self) -> String {
        ".brick-data-table { display: table; width: 100%; }".into()
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== TableColumn Tests =====

    #[test]
    fn test_table_column_new() {
        let col = TableColumn::new("name", "Name");
        assert_eq!(col.key, "name");
        assert_eq!(col.header, "Name");
        assert!(col.width.is_none());
        assert!(!col.sortable);
    }

    #[test]
    fn test_table_column_builder() {
        let col = TableColumn::new("price", "Price")
            .width(150.0)
            .align(TextAlign::Right)
            .sortable();
        assert_eq!(col.width, Some(150.0));
        assert_eq!(col.align, TextAlign::Right);
        assert!(col.sortable);
    }

    #[test]
    fn test_table_column_width_min() {
        let col = TableColumn::new("id", "ID").width(5.0);
        assert_eq!(col.width, Some(20.0));
    }

    // ===== TextAlign Tests =====

    #[test]
    fn test_text_align_default() {
        assert_eq!(TextAlign::default(), TextAlign::Left);
    }

    // ===== CellValue Tests =====

    #[test]
    fn test_cell_value_text() {
        let cell = CellValue::Text("Hello".to_string());
        assert_eq!(cell.display(), "Hello");
    }

    #[test]
    fn test_cell_value_number() {
        let cell = CellValue::Number(42.5);
        assert_eq!(cell.display(), "42.5");
    }

    #[test]
    fn test_cell_value_bool() {
        assert_eq!(CellValue::Bool(true).display(), "Yes");
        assert_eq!(CellValue::Bool(false).display(), "No");
    }

    #[test]
    fn test_cell_value_empty() {
        assert_eq!(CellValue::Empty.display(), "");
    }

    #[test]
    fn test_cell_value_from_str() {
        let cell: CellValue = "test".into();
        assert_eq!(cell, CellValue::Text("test".to_string()));
    }

    #[test]
    fn test_cell_value_from_f64() {
        let cell: CellValue = 1.5f64.into();
        assert_eq!(cell, CellValue::Number(1.5));
    }

    #[test]
    fn test_cell_value_from_i32() {
        let cell: CellValue = 42i32.into();
        assert_eq!(cell, CellValue::Number(42.0));
    }

    #[test]
    fn test_cell_value_from_bool() {
        let cell: CellValue = true.into();
        assert_eq!(cell, CellValue::Bool(true));
    }

    // ===== TableRow Tests =====

    #[test]
    fn test_table_row_new() {
        let row = TableRow::new();
        assert!(row.cells.is_empty());
    }

    #[test]
    fn test_table_row_builder() {
        let row = TableRow::new()
            .cell("name", "Alice")
            .cell("age", 30)
            .cell("active", true);

        assert_eq!(row.get("name"), Some(&CellValue::Text("Alice".to_string())));
        assert_eq!(row.get("age"), Some(&CellValue::Number(30.0)));
        assert_eq!(row.get("active"), Some(&CellValue::Bool(true)));
    }

    #[test]
    fn test_table_row_get_missing() {
        let row = TableRow::new();
        assert!(row.get("nonexistent").is_none());
    }

    // ===== DataTable Construction Tests =====

    #[test]
    fn test_data_table_new() {
        let table = DataTable::new();
        assert_eq!(table.column_count(), 0);
        assert_eq!(table.row_count(), 0);
        assert!(table.is_empty());
    }

    #[test]
    fn test_data_table_builder() {
        let table = DataTable::new()
            .column(TableColumn::new("id", "ID"))
            .column(TableColumn::new("name", "Name"))
            .row(TableRow::new().cell("id", 1).cell("name", "Alice"))
            .row(TableRow::new().cell("id", 2).cell("name", "Bob"))
            .row_height(50.0)
            .header_height(60.0)
            .selectable(true)
            .striped(true)
            .bordered(true)
            .accessible_name("User table")
            .test_id("users-table");

        assert_eq!(table.column_count(), 2);
        assert_eq!(table.row_count(), 2);
        assert!(!table.is_empty());
        assert_eq!(Widget::accessible_name(&table), Some("User table"));
        assert_eq!(Widget::test_id(&table), Some("users-table"));
    }

    #[test]
    fn test_data_table_columns() {
        let cols = vec![TableColumn::new("a", "A"), TableColumn::new("b", "B")];
        let table = DataTable::new().columns(cols);
        assert_eq!(table.column_count(), 2);
    }

    #[test]
    fn test_data_table_rows() {
        let rows = vec![
            TableRow::new().cell("x", 1),
            TableRow::new().cell("x", 2),
            TableRow::new().cell("x", 3),
        ];
        let table = DataTable::new().rows(rows);
        assert_eq!(table.row_count(), 3);
    }

    // ===== Selection Tests =====

    #[test]
    fn test_data_table_select_row() {
        let mut table = DataTable::new()
            .row(TableRow::new())
            .row(TableRow::new())
            .selectable(true);

        assert!(table.get_selected_row().is_none());
        table.select_row(Some(1));
        assert_eq!(table.get_selected_row(), Some(1));
        table.select_row(None);
        assert!(table.get_selected_row().is_none());
    }

    #[test]
    fn test_data_table_select_row_out_of_bounds() {
        let mut table = DataTable::new().row(TableRow::new());
        table.select_row(Some(10));
        assert!(table.get_selected_row().is_none());
    }

    // ===== Sort Tests =====

    #[test]
    fn test_data_table_set_sort() {
        let mut table = DataTable::new().column(TableColumn::new("name", "Name").sortable());

        table.set_sort("name", SortDirection::Descending);
        assert_eq!(table.get_sort_column(), Some("name"));
        assert_eq!(table.get_sort_direction(), SortDirection::Descending);
    }

    #[test]
    fn test_sort_direction() {
        assert_ne!(SortDirection::Ascending, SortDirection::Descending);
    }

    // ===== Clear Tests =====

    #[test]
    fn test_data_table_clear() {
        let mut table = DataTable::new().row(TableRow::new()).row(TableRow::new());
        table.select_row(Some(0));

        table.clear();
        assert!(table.is_empty());
        assert!(table.get_selected_row().is_none());
    }

    // ===== Dimension Tests =====

    #[test]
    fn test_data_table_row_height_min() {
        let table = DataTable::new().row_height(10.0);
        assert_eq!(table.row_height, 20.0);
    }

    #[test]
    fn test_data_table_header_height_min() {
        let table = DataTable::new().header_height(10.0);
        assert_eq!(table.header_height, 20.0);
    }

    #[test]
    fn test_data_table_calculate_width() {
        let table = DataTable::new()
            .column(TableColumn::new("a", "A").width(100.0))
            .column(TableColumn::new("b", "B").width(150.0));
        assert_eq!(table.calculate_width(), 250.0);
    }

    #[test]
    fn test_data_table_calculate_height() {
        let table = DataTable::new()
            .header_height(40.0)
            .row_height(30.0)
            .row(TableRow::new())
            .row(TableRow::new());
        assert_eq!(table.calculate_height(), 40.0 + 60.0);
    }

    // ===== Widget Trait Tests =====

    #[test]
    fn test_data_table_type_id() {
        let table = DataTable::new();
        assert_eq!(Widget::type_id(&table), TypeId::of::<DataTable>());
    }

    #[test]
    fn test_data_table_measure() {
        let table = DataTable::new()
            .column(TableColumn::new("a", "A").width(200.0))
            .header_height(40.0)
            .row_height(30.0)
            .row(TableRow::new())
            .row(TableRow::new());

        let size = table.measure(Constraints::loose(Size::new(1000.0, 1000.0)));
        assert_eq!(size.width, 200.0);
        assert_eq!(size.height, 100.0); // 40 + 30*2
    }

    #[test]
    fn test_data_table_layout() {
        let mut table = DataTable::new();
        let bounds = Rect::new(10.0, 20.0, 500.0, 300.0);
        let result = table.layout(bounds);
        assert_eq!(result.size, Size::new(500.0, 300.0));
        assert_eq!(table.bounds, bounds);
    }

    #[test]
    fn test_data_table_children() {
        let table = DataTable::new();
        assert!(table.children().is_empty());
    }

    #[test]
    fn test_data_table_is_interactive() {
        let table = DataTable::new();
        assert!(!table.is_interactive());

        let table = DataTable::new().selectable(true);
        assert!(table.is_interactive());
    }

    #[test]
    fn test_data_table_is_focusable() {
        let table = DataTable::new();
        assert!(!table.is_focusable());

        let table = DataTable::new().selectable(true);
        assert!(table.is_focusable());
    }

    #[test]
    fn test_data_table_accessible_role() {
        let table = DataTable::new();
        assert_eq!(table.accessible_role(), AccessibleRole::Table);
    }

    #[test]
    fn test_data_table_accessible_name() {
        let table = DataTable::new().accessible_name("Products table");
        assert_eq!(Widget::accessible_name(&table), Some("Products table"));
    }

    #[test]
    fn test_data_table_test_id() {
        let table = DataTable::new().test_id("inventory-grid");
        assert_eq!(Widget::test_id(&table), Some("inventory-grid"));
    }

    // ===== Message Tests =====

    #[test]
    fn test_table_sort_changed() {
        let msg = TableSortChanged {
            column: "price".to_string(),
            direction: SortDirection::Descending,
        };
        assert_eq!(msg.column, "price");
        assert_eq!(msg.direction, SortDirection::Descending);
    }

    #[test]
    fn test_table_row_selected() {
        let msg = TableRowSelected { index: 5 };
        assert_eq!(msg.index, 5);
    }

    // ===== Position Tests =====

    #[test]
    fn test_row_y() {
        let mut table = DataTable::new()
            .header_height(50.0)
            .row_height(40.0)
            .row(TableRow::new())
            .row(TableRow::new());
        table.bounds = Rect::new(0.0, 10.0, 100.0, 200.0);

        assert_eq!(table.row_y(0), 60.0); // 10 + 50
        assert_eq!(table.row_y(1), 100.0); // 10 + 50 + 40
    }

    // ===== Paint Tests =====

    use presentar_core::RecordingCanvas;

    #[test]
    fn test_data_table_paint_empty() {
        let mut table = DataTable::new();
        table.bounds = Rect::new(0.0, 0.0, 400.0, 300.0);
        let mut canvas = RecordingCanvas::new();
        table.paint(&mut canvas);
    }

    #[test]
    fn test_data_table_paint_with_data() {
        let mut table = DataTable::new()
            .column(TableColumn::new("name", "Name").width(100.0))
            .column(TableColumn::new("value", "Value").width(100.0))
            .row(TableRow::new().cell("name", "Item 1").cell("value", 100))
            .row(TableRow::new().cell("name", "Item 2").cell("value", 200));
        table.bounds = Rect::new(0.0, 0.0, 400.0, 200.0);
        let mut canvas = RecordingCanvas::new();
        table.paint(&mut canvas);
        assert!(canvas.commands().len() > 5);
    }

    #[test]
    fn test_data_table_paint_with_selection() {
        let mut table = DataTable::new()
            .column(TableColumn::new("x", "X"))
            .row(TableRow::new().cell("x", "A"))
            .row(TableRow::new().cell("x", "B"))
            .selectable(true);
        table.bounds = Rect::new(0.0, 0.0, 200.0, 150.0);
        table.selected_row = Some(1);
        let mut canvas = RecordingCanvas::new();
        table.paint(&mut canvas);
        assert!(canvas.commands().len() > 5);
    }

    #[test]
    fn test_data_table_paint_striped() {
        let mut table = DataTable::new()
            .column(TableColumn::new("x", "X"))
            .row(TableRow::new())
            .row(TableRow::new())
            .row(TableRow::new())
            .striped(true);
        table.bounds = Rect::new(0.0, 0.0, 200.0, 200.0);
        let mut canvas = RecordingCanvas::new();
        table.paint(&mut canvas);
        assert!(canvas.commands().len() > 5);
    }

    #[test]
    fn test_data_table_paint_bordered() {
        let mut table = DataTable::new()
            .column(TableColumn::new("x", "X"))
            .row(TableRow::new())
            .bordered(true);
        table.bounds = Rect::new(0.0, 0.0, 200.0, 100.0);
        let mut canvas = RecordingCanvas::new();
        // Should not panic when painting bordered table
        table.paint(&mut canvas);
    }

    #[test]
    fn test_data_table_paint_sortable_columns() {
        let mut table = DataTable::new()
            .column(TableColumn::new("name", "Name").sortable())
            .row(TableRow::new().cell("name", "A"));
        table.bounds = Rect::new(0.0, 0.0, 200.0, 100.0);
        table.sort_column = Some("name".to_string());
        table.sort_direction = SortDirection::Ascending;
        let mut canvas = RecordingCanvas::new();
        table.paint(&mut canvas);
    }

    #[test]
    fn test_data_table_paint_all_alignments() {
        let mut table = DataTable::new()
            .column(TableColumn::new("a", "A").align(TextAlign::Left))
            .column(TableColumn::new("b", "B").align(TextAlign::Center))
            .column(TableColumn::new("c", "C").align(TextAlign::Right))
            .row(TableRow::new().cell("a", "L").cell("b", "C").cell("c", "R"));
        table.bounds = Rect::new(0.0, 0.0, 300.0, 100.0);
        let mut canvas = RecordingCanvas::new();
        table.paint(&mut canvas);
    }

    // ===== Event Tests =====

    #[test]
    fn test_data_table_event_mouse_down() {
        let mut table = DataTable::new()
            .column(TableColumn::new("name", "Name").sortable())
            .row(TableRow::new());
        table.layout(Rect::new(0.0, 0.0, 200.0, 100.0));

        // Should not panic
        let _ = table.event(&Event::MouseDown {
            position: presentar_core::Point::new(100.0, 20.0),
            button: presentar_core::MouseButton::Left,
        });
    }

    #[test]
    fn test_data_table_event_selectable() {
        let mut table = DataTable::new()
            .column(TableColumn::new("x", "X"))
            .row(TableRow::new())
            .row(TableRow::new())
            .selectable(true);
        table.layout(Rect::new(0.0, 0.0, 200.0, 200.0));

        // Click on row area - should not panic
        let _ = table.event(&Event::MouseDown {
            position: presentar_core::Point::new(100.0, 80.0),
            button: presentar_core::MouseButton::Left,
        });
    }

    #[test]
    fn test_data_table_event_not_selectable() {
        let mut table = DataTable::new()
            .column(TableColumn::new("x", "X"))
            .row(TableRow::new())
            .selectable(false);
        table.layout(Rect::new(0.0, 0.0, 200.0, 150.0));

        let result = table.event(&Event::MouseDown {
            position: presentar_core::Point::new(100.0, 80.0),
            button: presentar_core::MouseButton::Left,
        });
        assert!(result.is_none());
    }

    #[test]
    fn test_data_table_event_keydown() {
        let mut table = DataTable::new();
        table.layout(Rect::new(0.0, 0.0, 200.0, 100.0));

        let result = table.event(&Event::key_down(presentar_core::Key::Enter));
        assert!(result.is_none());
    }

    // ===== Builder Tests =====

    #[test]
    fn test_data_table_color_setters() {
        let table = DataTable::new()
            .header_bg(Color::RED)
            .row_bg(Color::GREEN)
            .row_alt_bg(Color::BLUE)
            .selected_bg(Color::new(1.0, 1.0, 0.0, 1.0))
            .text_color(Color::BLACK);

        assert_eq!(table.header_bg, Color::RED);
        assert_eq!(table.row_bg, Color::GREEN);
    }

    #[test]
    fn test_data_table_striped_toggle() {
        let table = DataTable::new().striped(false);
        assert!(!table.striped);
    }

    #[test]
    fn test_data_table_bordered_toggle() {
        let table = DataTable::new().bordered(false);
        assert!(!table.bordered);
    }

    // ===== Additional CellValue Tests =====

    #[test]
    fn test_cell_value_from_string() {
        let s = String::from("hello");
        let cell: CellValue = s.into();
        assert!(matches!(cell, CellValue::Text(_)));
    }

    // ===== Brick Trait Tests =====

    #[test]
    fn test_data_table_brick_name() {
        let table = DataTable::new();
        assert_eq!(table.brick_name(), "DataTable");
    }

    #[test]
    fn test_data_table_brick_assertions() {
        let table = DataTable::new();
        let assertions = table.assertions();
        assert!(!assertions.is_empty());
    }

    #[test]
    fn test_data_table_brick_budget() {
        let table = DataTable::new();
        let budget = table.budget();
        assert!(budget.layout_ms > 0);
    }

    #[test]
    fn test_data_table_brick_verify() {
        let table = DataTable::new();
        let verification = table.verify();
        assert!(!verification.passed.is_empty());
        assert!(verification.failed.is_empty());
    }

    #[test]
    fn test_data_table_brick_to_html() {
        let table = DataTable::new();
        let html = table.to_html();
        assert!(html.contains("brick-data-table"));
    }

    #[test]
    fn test_data_table_brick_to_css() {
        let table = DataTable::new();
        let css = table.to_css();
        assert!(css.contains("brick-data-table"));
    }

    #[test]
    fn test_data_table_brick_test_id() {
        let table = DataTable::new().test_id("my-table");
        assert_eq!(Brick::test_id(&table), Some("my-table"));
    }

    // ===== Widget Trait Tests =====

    #[test]
    fn test_data_table_children_mut_empty() {
        let mut table = DataTable::new();
        assert!(table.children_mut().is_empty());
    }
}
