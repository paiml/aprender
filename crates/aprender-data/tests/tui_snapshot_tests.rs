//! TUI Snapshot Tests
//!
//! Tests for TUI rendering using jugar-probar style snapshot testing.
//! These tests capture the rendered output and verify it matches expectations.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::uninlined_format_args,
    clippy::suboptimal_flops
)]

use std::sync::Arc;

use alimentar::tui::{DatasetAdapter, DatasetViewer, RowDetailView, SchemaInspector, ScrollState};
use arrow::{
    array::{Float32Array, Int32Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

// ============================================================================
// Test Helpers
// ============================================================================

fn create_test_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Int32, false),
        Field::new("score", DataType::Float32, true),
    ]))
}

fn create_test_batch(schema: &Arc<Schema>, count: usize) -> RecordBatch {
    let ids: Vec<String> = (0..count).map(|i| format!("id_{}", i)).collect();
    let names: Vec<String> = (0..count)
        .map(|i| format!("Item {} Description", i))
        .collect();
    let values: Vec<i32> = (0..count).map(|i| (i * 100) as i32).collect();
    let scores: Vec<f32> = (0..count).map(|i| (i as f32) * 0.1 + 0.5).collect();

    RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(Int32Array::from(values)),
            Arc::new(Float32Array::from(scores)),
        ],
    )
    .unwrap()
}

fn create_test_adapter(rows: usize) -> DatasetAdapter {
    let schema = create_test_schema();
    let batch = create_test_batch(&schema, rows);
    DatasetAdapter::from_batches(vec![batch], schema).unwrap()
}

// ============================================================================
// DatasetViewer Snapshot Tests
// ============================================================================

#[test]
fn test_viewer_render_header() {
    let adapter = create_test_adapter(10);
    let viewer = DatasetViewer::with_dimensions(adapter, 80, 10);

    let header = viewer.render_header_line();

    // Header should contain column names
    assert!(header.contains("id"));
    assert!(header.contains("name"));
    assert!(header.contains("value"));
    assert!(header.contains("score"));
}

#[test]
fn test_viewer_render_first_row() {
    let adapter = create_test_adapter(10);
    let viewer = DatasetViewer::with_dimensions(adapter, 80, 10);

    let row = viewer.render_row_line(0);
    assert!(row.is_some());

    let row_text = row.unwrap();
    assert!(row_text.contains("id_0"));
}

#[test]
fn test_viewer_render_lines_count() {
    let adapter = create_test_adapter(5);
    let viewer = DatasetViewer::with_dimensions(adapter, 80, 10);

    let lines = viewer.render_lines();

    // Should have header + 5 data rows
    assert_eq!(lines.len(), 6);
}

#[test]
fn test_viewer_visible_rows_data() {
    let adapter = create_test_adapter(20);
    let mut viewer = DatasetViewer::with_dimensions(adapter, 80, 10);

    // Initially at offset 0
    let visible = viewer.visible_rows_data();
    assert!(!visible.is_empty());
    assert!(visible.len() <= 9); // visible_rows = height - 1

    // Scroll down
    viewer.scroll_down();
    viewer.scroll_down();
    let visible_after = viewer.visible_rows_data();
    assert!(!visible_after.is_empty());
}

#[test]
fn test_viewer_scroll_operations() {
    let adapter = create_test_adapter(50);
    let mut viewer = DatasetViewer::with_dimensions(adapter, 80, 10);

    assert_eq!(viewer.scroll_offset(), 0);

    viewer.page_down();
    assert!(viewer.scroll_offset() > 0);

    let offset_after_page = viewer.scroll_offset();
    viewer.page_up();
    assert!(viewer.scroll_offset() < offset_after_page);

    viewer.end();
    let end_offset = viewer.scroll_offset();
    assert!(end_offset > 0);

    viewer.home();
    assert_eq!(viewer.scroll_offset(), 0);
}

#[test]
fn test_viewer_selection_operations() {
    let adapter = create_test_adapter(20);
    let mut viewer = DatasetViewer::with_dimensions(adapter, 80, 10);

    assert!(viewer.selected_row().is_none());

    viewer.select_row(5);
    assert_eq!(viewer.selected_row(), Some(5));
    assert!(viewer.is_row_selected(5));
    assert!(!viewer.is_row_selected(4));

    viewer.select_next();
    assert_eq!(viewer.selected_row(), Some(6));

    viewer.select_prev();
    assert_eq!(viewer.selected_row(), Some(5));

    viewer.clear_selection();
    assert!(viewer.selected_row().is_none());
}

#[test]
fn test_viewer_search_functionality() {
    let adapter = create_test_adapter(20);
    let mut viewer = DatasetViewer::with_dimensions(adapter, 80, 10);

    // Search for a specific ID
    let result = viewer.search("id_15");
    assert_eq!(result, Some(15));
    assert_eq!(viewer.selected_row(), Some(15));

    // Search next
    let next = viewer.search_next("id_");
    assert!(next.is_some());
    // Should find next row after 15
    assert!(next.unwrap() > 15 || next.unwrap() == 0);
}

#[test]
fn test_viewer_scrollbar_info() {
    let adapter = create_test_adapter(50);
    let mut viewer = DatasetViewer::with_dimensions(adapter, 80, 10);

    assert!(viewer.needs_scrollbar());

    let size = viewer.scrollbar_size();
    assert!(size > 0.0 && size < 1.0);

    let pos_start = viewer.scrollbar_position();
    assert!((pos_start - 0.0).abs() < 0.01);

    viewer.end();
    let pos_end = viewer.scrollbar_position();
    assert!(pos_end > 0.5);
}

#[test]
fn test_viewer_no_scrollbar_small_dataset() {
    let adapter = create_test_adapter(3);
    let viewer = DatasetViewer::with_dimensions(adapter, 80, 20);

    assert!(!viewer.needs_scrollbar());
    assert!((viewer.scrollbar_size() - 1.0).abs() < 0.01);
}

// ============================================================================
// SchemaInspector Snapshot Tests
// ============================================================================

#[test]
fn test_schema_inspector_render() {
    let adapter = create_test_adapter(10);
    let inspector = SchemaInspector::new(&adapter);

    assert_eq!(inspector.field_count(), 4);

    let lines = inspector.render_lines();

    // Header + separator + 4 fields
    assert_eq!(lines.len(), 6);

    // Check header
    assert!(lines[0].contains("Field"));
    assert!(lines[0].contains("Type"));
    assert!(lines[0].contains("Nullable"));

    // Check separator
    assert!(lines[1].contains("---"));

    // Check fields
    assert!(lines[2].contains("id"));
    assert!(lines[3].contains("name"));
    assert!(lines[4].contains("value"));
    assert!(lines[5].contains("score"));
}

#[test]
fn test_schema_inspector_field_info() {
    let adapter = create_test_adapter(10);
    let inspector = SchemaInspector::new(&adapter);

    let (name, type_name, nullable) = inspector.field(0).unwrap();
    assert_eq!(name, "id");
    assert_eq!(type_name, "Utf8");
    assert!(!nullable);

    let (name, _, nullable) = inspector.field(3).unwrap();
    assert_eq!(name, "score");
    assert!(nullable);
}

#[test]
fn test_schema_inspector_field_names_and_types() {
    let adapter = create_test_adapter(10);
    let inspector = SchemaInspector::new(&adapter);

    let names = inspector.field_names();
    assert_eq!(names, vec!["id", "name", "value", "score"]);

    let types = inspector.type_names();
    assert_eq!(types[0], "Utf8");
    assert_eq!(types[2], "Int32");
    assert_eq!(types[3], "Float32");
}

// ============================================================================
// RowDetailView Snapshot Tests
// ============================================================================

#[test]
fn test_row_detail_view_render() {
    let adapter = create_test_adapter(10);
    let detail = RowDetailView::new(&adapter, 5).unwrap();

    assert_eq!(detail.row_index(), 5);
    assert_eq!(detail.field_count(), 4);

    let lines = detail.render_lines();
    assert!(!lines.is_empty());
    assert!(lines[0].contains("Row 5"));
}

#[test]
fn test_row_detail_view_field_values() {
    let adapter = create_test_adapter(10);
    let detail = RowDetailView::new(&adapter, 3).unwrap();

    let id_value = detail.field_value("id");
    assert_eq!(id_value, Some("id_3"));

    let value_value = detail.field_value("value");
    assert_eq!(value_value, Some("300"));

    let (name, value) = detail.field_by_index(0).unwrap();
    assert_eq!(name, "id");
    assert_eq!(value, "id_3");
}

#[test]
fn test_row_detail_view_scroll() {
    let adapter = create_test_adapter(10);
    let mut detail = RowDetailView::with_dimensions(&adapter, 0, 40, 5).unwrap();

    let initial_offset = detail.scroll_offset();
    detail.scroll_down();
    detail.scroll_down();
    assert!(detail.scroll_offset() >= initial_offset);

    detail.scroll_up();
    // Offset may or may not change depending on content

    detail.page_down();
    let after_page_down = detail.scroll_offset();
    detail.page_up();
    assert!(detail.scroll_offset() <= after_page_down);
}

#[test]
fn test_row_detail_view_render_string() {
    let adapter = create_test_adapter(10);
    let detail = RowDetailView::new(&adapter, 2).unwrap();

    let rendered = detail.render();
    assert!(rendered.contains("Row 2"));
    assert!(rendered.contains("id:"));
    assert!(rendered.contains("id_2"));
}

#[test]
fn test_row_detail_view_out_of_bounds() {
    let adapter = create_test_adapter(10);
    let detail = RowDetailView::new(&adapter, 100);
    assert!(detail.is_none());
}

// ============================================================================
// ScrollState Tests
// ============================================================================

#[test]
fn test_scroll_state_navigation() {
    let mut scroll = ScrollState::new(100, 20);

    assert_eq!(scroll.offset(), 0);
    assert_eq!(scroll.total_rows(), 100);
    assert_eq!(scroll.visible_rows(), 20);

    scroll.scroll_down();
    assert_eq!(scroll.offset(), 1);

    scroll.scroll_up();
    assert_eq!(scroll.offset(), 0);

    scroll.scroll_up(); // Should stay at 0
    assert_eq!(scroll.offset(), 0);
}

#[test]
fn test_scroll_state_paging() {
    let mut scroll = ScrollState::new(100, 20);

    scroll.page_down();
    assert_eq!(scroll.offset(), 20);

    scroll.page_down();
    assert_eq!(scroll.offset(), 40);

    scroll.page_up();
    assert_eq!(scroll.offset(), 20);

    scroll.home();
    assert_eq!(scroll.offset(), 0);

    scroll.end();
    assert_eq!(scroll.offset(), 80); // 100 - 20
}

#[test]
fn test_scroll_state_selection() {
    let mut scroll = ScrollState::new(100, 20);

    scroll.set_selected(Some(50));
    assert_eq!(scroll.selected(), Some(50));

    scroll.select_next();
    assert_eq!(scroll.selected(), Some(51));

    scroll.select_prev();
    assert_eq!(scroll.selected(), Some(50));

    scroll.set_selected(None);
    assert!(scroll.selected().is_none());

    scroll.select_next();
    assert_eq!(scroll.selected(), Some(0));
}

#[test]
fn test_scroll_state_visibility() {
    let mut scroll = ScrollState::new(100, 20);
    scroll.set_offset(30);

    assert!(scroll.is_visible(35));
    assert!(!scroll.is_visible(29));
    assert!(!scroll.is_visible(50));

    assert_eq!(scroll.to_viewport_row(35), Some(5));
    assert_eq!(scroll.to_viewport_row(29), None);

    assert_eq!(scroll.to_global_row(5), 35);
}

#[test]
fn test_scroll_state_ensure_visible() {
    let mut scroll = ScrollState::new(100, 20);

    scroll.ensure_visible(50);
    assert!(scroll.is_visible(50));

    scroll.ensure_visible(10);
    assert!(scroll.is_visible(10));
}

#[test]
fn test_scroll_state_update_total_rows() {
    let mut scroll = ScrollState::new(100, 20);
    scroll.set_offset(80);
    scroll.set_selected(Some(90));

    scroll.set_total_rows(50);

    // Offset and selection should be clamped
    assert!(scroll.offset() <= 30);
    assert!(scroll.selected().unwrap_or(0) < 50);
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_full_viewer_workflow() {
    let adapter = create_test_adapter(100);
    let mut viewer = DatasetViewer::with_dimensions(adapter, 80, 24);

    // Initial state
    assert_eq!(viewer.row_count(), 100);
    assert!(!viewer.is_empty());

    // Render initial view
    let lines = viewer.render_lines();
    assert!(lines[0].contains("id"));

    // Navigate
    viewer.page_down();
    viewer.select_row(30);
    assert_eq!(viewer.selected_row(), Some(30));

    // Search
    let found = viewer.search("id_75");
    assert_eq!(found, Some(75));

    // Resize
    viewer.set_dimensions(40, 12);
    let narrow_lines = viewer.render_lines();
    assert!(!narrow_lines.is_empty());
}

#[test]
fn test_empty_dataset_handling() {
    let schema = create_test_schema();
    let adapter = DatasetAdapter::from_batches(vec![], schema).unwrap();
    let viewer = DatasetViewer::new(adapter);

    assert!(viewer.is_empty());
    assert_eq!(viewer.row_count(), 0);
    assert!(!viewer.needs_scrollbar());

    let lines = viewer.render_lines();
    // Should have at least the header
    assert!(!lines.is_empty());
}

#[test]
fn test_streaming_mode_operations() {
    let schema = create_test_schema();
    let batch = create_test_batch(&schema, 50);
    let adapter = DatasetAdapter::streaming_from_batches(vec![batch], schema).unwrap();

    assert!(adapter.is_streaming());
    assert_eq!(adapter.row_count(), 50);

    let viewer = DatasetViewer::new(adapter);
    let lines = viewer.render_lines();
    assert!(!lines.is_empty());
}
