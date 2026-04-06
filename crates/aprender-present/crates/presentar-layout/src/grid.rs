//! CSS Grid-like layout system.
//!
//! This module provides a grid layout system similar to CSS Grid,
//! supporting:
//! - Fixed and flexible track sizes (px, fr units)
//! - Row and column spans
//! - Gaps between tracks
//! - Named grid areas
//! - Auto-placement

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A track size specification.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TrackSize {
    /// Fixed size in pixels
    Px(f32),
    /// Flexible fraction of remaining space
    Fr(f32),
    /// Size based on content
    Auto,
    /// Minimum content size
    MinContent,
    /// Maximum content size
    MaxContent,
}

impl Default for TrackSize {
    fn default() -> Self {
        Self::Fr(1.0)
    }
}

impl TrackSize {
    /// Create a fixed pixel size.
    #[must_use]
    pub const fn px(value: f32) -> Self {
        Self::Px(value)
    }

    /// Create a flexible fraction.
    #[must_use]
    pub const fn fr(value: f32) -> Self {
        Self::Fr(value)
    }

    /// Auto size.
    pub const AUTO: Self = Self::Auto;
}

/// Grid template definition.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GridTemplate {
    /// Column track sizes
    pub columns: Vec<TrackSize>,
    /// Row track sizes
    pub rows: Vec<TrackSize>,
    /// Gap between columns
    pub column_gap: f32,
    /// Gap between rows
    pub row_gap: f32,
    /// Named grid areas (row-major order)
    pub areas: HashMap<String, GridArea>,
}

impl GridTemplate {
    /// Create a new empty grid template.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a grid with specified columns and default 1fr rows.
    #[must_use]
    pub fn columns(cols: impl IntoIterator<Item = TrackSize>) -> Self {
        Self {
            columns: cols.into_iter().collect(),
            ..Self::default()
        }
    }

    /// Create a 12-column grid (common dashboard layout).
    #[must_use]
    pub fn twelve_column() -> Self {
        Self {
            columns: vec![TrackSize::Fr(1.0); 12],
            column_gap: 16.0,
            row_gap: 16.0,
            ..Self::default()
        }
    }

    /// Set row track sizes.
    #[must_use]
    pub fn with_rows(mut self, rows: impl IntoIterator<Item = TrackSize>) -> Self {
        self.rows = rows.into_iter().collect();
        self
    }

    /// Set column gap.
    #[must_use]
    pub const fn with_column_gap(mut self, gap: f32) -> Self {
        self.column_gap = gap;
        self
    }

    /// Set row gap.
    #[must_use]
    pub const fn with_row_gap(mut self, gap: f32) -> Self {
        self.row_gap = gap;
        self
    }

    /// Set both gaps.
    #[must_use]
    pub const fn with_gap(mut self, gap: f32) -> Self {
        self.column_gap = gap;
        self.row_gap = gap;
        self
    }

    /// Add a named area.
    #[must_use]
    pub fn with_area(mut self, name: impl Into<String>, area: GridArea) -> Self {
        self.areas.insert(name.into(), area);
        self
    }

    /// Get number of columns.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Get number of explicit rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
}

/// A named grid area spanning rows and columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridArea {
    /// Starting row (0-indexed)
    pub row_start: usize,
    /// Ending row (exclusive)
    pub row_end: usize,
    /// Starting column (0-indexed)
    pub col_start: usize,
    /// Ending column (exclusive)
    pub col_end: usize,
}

impl GridArea {
    /// Create a new grid area.
    #[must_use]
    pub const fn new(row_start: usize, col_start: usize, row_end: usize, col_end: usize) -> Self {
        Self {
            row_start,
            row_end,
            col_start,
            col_end,
        }
    }

    /// Create a single-cell area.
    #[must_use]
    pub const fn cell(row: usize, col: usize) -> Self {
        Self {
            row_start: row,
            row_end: row + 1,
            col_start: col,
            col_end: col + 1,
        }
    }

    /// Create an area spanning multiple columns in a single row.
    #[must_use]
    pub const fn row_span(row: usize, col_start: usize, col_end: usize) -> Self {
        Self {
            row_start: row,
            row_end: row + 1,
            col_start,
            col_end,
        }
    }

    /// Create an area spanning multiple rows in a single column.
    #[must_use]
    pub const fn col_span(col: usize, row_start: usize, row_end: usize) -> Self {
        Self {
            row_start,
            row_end,
            col_start: col,
            col_end: col + 1,
        }
    }

    /// Get the number of rows this area spans.
    #[must_use]
    pub const fn row_span_count(&self) -> usize {
        self.row_end.saturating_sub(self.row_start)
    }

    /// Get the number of columns this area spans.
    #[must_use]
    pub const fn col_span_count(&self) -> usize {
        self.col_end.saturating_sub(self.col_start)
    }
}

/// Grid item placement properties.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GridItem {
    /// Column start (1-indexed like CSS, 0 = auto)
    pub column_start: usize,
    /// Column end (exclusive, 0 = auto)
    pub column_end: usize,
    /// Row start (1-indexed like CSS, 0 = auto)
    pub row_start: usize,
    /// Row end (exclusive, 0 = auto)
    pub row_end: usize,
    /// Named area to place in
    pub area: Option<String>,
    /// Column span (alternative to column_end)
    pub column_span: usize,
    /// Row span (alternative to row_end)
    pub row_span: usize,
    /// Alignment within cell (horizontal)
    pub justify_self: Option<GridAlign>,
    /// Alignment within cell (vertical)
    pub align_self: Option<GridAlign>,
}

impl GridItem {
    /// Create a new grid item with auto placement.
    #[must_use]
    pub fn new() -> Self {
        Self {
            column_span: 1,
            row_span: 1,
            ..Self::default()
        }
    }

    /// Place in a specific column.
    #[must_use]
    pub const fn column(mut self, col: usize) -> Self {
        self.column_start = col;
        self.column_end = col + 1;
        self
    }

    /// Place in a specific row.
    #[must_use]
    pub const fn row(mut self, row: usize) -> Self {
        self.row_start = row;
        self.row_end = row + 1;
        self
    }

    /// Span multiple columns.
    #[must_use]
    pub const fn span_columns(mut self, span: usize) -> Self {
        self.column_span = span;
        self
    }

    /// Span multiple rows.
    #[must_use]
    pub const fn span_rows(mut self, span: usize) -> Self {
        self.row_span = span;
        self
    }

    /// Place in a named area.
    #[must_use]
    pub fn in_area(mut self, area: impl Into<String>) -> Self {
        self.area = Some(area.into());
        self
    }

    /// Set horizontal alignment.
    #[must_use]
    pub const fn justify_self(mut self, align: GridAlign) -> Self {
        self.justify_self = Some(align);
        self
    }

    /// Set vertical alignment.
    #[must_use]
    pub const fn align_self(mut self, align: GridAlign) -> Self {
        self.align_self = Some(align);
        self
    }

    /// Get effective column span.
    #[must_use]
    pub fn effective_column_span(&self) -> usize {
        if self.column_end > self.column_start {
            self.column_end - self.column_start
        } else {
            self.column_span.max(1)
        }
    }

    /// Get effective row span.
    #[must_use]
    pub fn effective_row_span(&self) -> usize {
        if self.row_end > self.row_start {
            self.row_end - self.row_start
        } else {
            self.row_span.max(1)
        }
    }
}

/// Alignment within a grid cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GridAlign {
    /// Align to start
    Start,
    /// Align to end
    End,
    /// Center
    #[default]
    Center,
    /// Stretch to fill
    Stretch,
}

/// Auto-placement flow direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GridAutoFlow {
    /// Fill rows first
    #[default]
    Row,
    /// Fill columns first
    Column,
    /// Row with dense packing
    RowDense,
    /// Column with dense packing
    ColumnDense,
}

/// Computed grid layout result.
#[derive(Debug, Clone, Default)]
pub struct GridLayout {
    /// Computed column positions and sizes (start, size)
    pub columns: Vec<(f32, f32)>,
    /// Computed row positions and sizes (start, size)
    pub rows: Vec<(f32, f32)>,
    /// Total grid width
    pub width: f32,
    /// Total grid height
    pub height: f32,
}

impl GridLayout {
    /// Get the bounds for a grid area.
    #[must_use]
    pub fn area_bounds(&self, area: &GridArea) -> Option<(f32, f32, f32, f32)> {
        if area.col_start >= self.columns.len() || area.row_start >= self.rows.len() {
            return None;
        }

        let col_start = area.col_start;
        let col_end = area.col_end.min(self.columns.len());
        let row_start = area.row_start;
        let row_end = area.row_end.min(self.rows.len());

        let x = self.columns.get(col_start).map(|(pos, _)| *pos)?;
        let y = self.rows.get(row_start).map(|(pos, _)| *pos)?;

        let width: f32 = self.columns[col_start..col_end]
            .iter()
            .map(|(_, size)| size)
            .sum();
        let height: f32 = self.rows[row_start..row_end]
            .iter()
            .map(|(_, size)| size)
            .sum();

        Some((x, y, width, height))
    }

    /// Get the bounds for a grid item.
    #[must_use]
    pub fn item_bounds(
        &self,
        item: &GridItem,
        row: usize,
        col: usize,
    ) -> Option<(f32, f32, f32, f32)> {
        let area = GridArea::new(
            row,
            col,
            row + item.effective_row_span(),
            col + item.effective_column_span(),
        );
        self.area_bounds(&area)
    }
}

/// Compute grid track sizes.
pub fn compute_grid_layout(
    template: &GridTemplate,
    available_width: f32,
    available_height: f32,
    child_sizes: &[(f32, f32)],
) -> GridLayout {
    // Calculate column sizes
    let columns = compute_track_sizes(
        &template.columns,
        available_width,
        template.column_gap,
        child_sizes
            .iter()
            .map(|(w, _)| *w)
            .collect::<Vec<_>>()
            .as_slice(),
    );

    // Determine row count from items if not specified
    let row_count = if template.rows.is_empty() {
        // Auto-generate rows based on number of items and columns
        let col_count = template.columns.len().max(1);
        (child_sizes.len() + col_count - 1) / col_count
    } else {
        template.rows.len()
    };

    // Calculate row sizes
    let row_templates: Vec<TrackSize> = if template.rows.is_empty() {
        vec![TrackSize::Auto; row_count]
    } else {
        template.rows.clone()
    };

    let rows = compute_track_sizes(
        &row_templates,
        available_height,
        template.row_gap,
        child_sizes
            .iter()
            .map(|(_, h)| *h)
            .collect::<Vec<_>>()
            .as_slice(),
    );

    let width = columns.last().map(|(pos, size)| pos + size).unwrap_or(0.0);
    let height = rows.last().map(|(pos, size)| pos + size).unwrap_or(0.0);

    GridLayout {
        columns,
        rows,
        width,
        height,
    }
}

/// Compute track sizes (used for both rows and columns).
fn compute_track_sizes(
    tracks: &[TrackSize],
    available: f32,
    gap: f32,
    content_sizes: &[f32],
) -> Vec<(f32, f32)> {
    if tracks.is_empty() {
        return Vec::new();
    }

    let track_count = tracks.len();
    let total_gap = gap * (track_count.saturating_sub(1)) as f32;
    let available_for_tracks = (available - total_gap).max(0.0);

    // First pass: compute fixed and auto sizes
    let mut sizes: Vec<f32> = Vec::with_capacity(track_count);
    let mut total_fixed = 0.0;
    let mut total_fr = 0.0;

    for (i, track) in tracks.iter().enumerate() {
        match track {
            TrackSize::Px(px) => {
                sizes.push(*px);
                total_fixed += px;
            }
            TrackSize::Fr(fr) => {
                sizes.push(0.0); // Placeholder
                total_fr += fr;
            }
            TrackSize::Auto | TrackSize::MinContent | TrackSize::MaxContent => {
                let content_size = content_sizes.get(i).copied().unwrap_or(0.0);
                sizes.push(content_size);
                total_fixed += content_size;
            }
        }
    }

    // Second pass: distribute flexible space
    let remaining = (available_for_tracks - total_fixed).max(0.0);
    if total_fr > 0.0 {
        for (i, track) in tracks.iter().enumerate() {
            if let TrackSize::Fr(fr) = track {
                sizes[i] = remaining * fr / total_fr;
            }
        }
    }

    // Convert to positions
    let mut result = Vec::with_capacity(track_count);
    let mut position = 0.0;

    for (i, &size) in sizes.iter().enumerate() {
        result.push((position, size));
        position += size;
        if i < track_count - 1 {
            position += gap;
        }
    }

    result
}

/// Get explicit placement from item's row/column or named area.
fn get_explicit_position(item: &GridItem, template: &GridTemplate) -> Option<(usize, usize)> {
    if item.column_start > 0 && item.row_start > 0 {
        return Some((item.row_start - 1, item.column_start - 1));
    }
    item.area
        .as_ref()
        .and_then(|name| template.areas.get(name))
        .map(|area| (area.row_start, area.col_start))
}

/// Mark grid cells as occupied.
fn mark_occupied(
    occupied: &mut Vec<Vec<bool>>,
    row: usize,
    col: usize,
    row_span: usize,
    col_span: usize,
    col_count: usize,
) {
    ensure_rows(occupied, row + row_span, col_count);
    for r in row..(row + row_span) {
        for c in col..(col + col_span).min(col_count) {
            occupied[r][c] = true;
        }
    }
}

/// Auto-place items in a grid.
#[must_use]
pub fn auto_place_items(
    template: &GridTemplate,
    items: &[GridItem],
    flow: GridAutoFlow,
) -> Vec<(usize, usize)> {
    let col_count = template.columns.len().max(1);
    let mut occupied: Vec<Vec<bool>> = Vec::new();
    let mut placements = Vec::with_capacity(items.len());

    for item in items {
        // Check for explicit placement
        if let Some(pos) = get_explicit_position(item, template) {
            placements.push(pos);
            continue;
        }

        // Auto-place
        let col_span = item.effective_column_span();
        let row_span = item.effective_row_span();

        let (row, col) = match flow {
            GridAutoFlow::Row | GridAutoFlow::RowDense => {
                find_next_position_row(&mut occupied, col_count, col_span, row_span)
            }
            GridAutoFlow::Column | GridAutoFlow::ColumnDense => {
                find_next_position_column(&mut occupied, col_count, col_span, row_span)
            }
        };

        mark_occupied(&mut occupied, row, col, row_span, col_span, col_count);
        placements.push((row, col));
    }

    placements
}

fn find_next_position_row(
    occupied: &mut Vec<Vec<bool>>,
    col_count: usize,
    col_span: usize,
    row_span: usize,
) -> (usize, usize) {
    let mut row = 0;
    loop {
        ensure_rows(occupied, row + row_span, col_count);
        for col in 0..=(col_count.saturating_sub(col_span)) {
            if can_place(occupied, row, col, row_span, col_span) {
                return (row, col);
            }
        }
        row += 1;
    }
}

fn find_next_position_column(
    occupied: &mut Vec<Vec<bool>>,
    col_count: usize,
    col_span: usize,
    row_span: usize,
) -> (usize, usize) {
    for col in 0..=(col_count.saturating_sub(col_span)) {
        let mut row = 0;
        loop {
            ensure_rows(occupied, row + row_span, col_count);
            if can_place(occupied, row, col, row_span, col_span) {
                return (row, col);
            }
            row += 1;
            if row > 1000 {
                break; // Safety limit
            }
        }
    }
    (0, 0) // Fallback
}

fn ensure_rows(occupied: &mut Vec<Vec<bool>>, min_rows: usize, col_count: usize) {
    while occupied.len() < min_rows {
        occupied.push(vec![false; col_count]);
    }
}

fn can_place(
    occupied: &[Vec<bool>],
    row: usize,
    col: usize,
    row_span: usize,
    col_span: usize,
) -> bool {
    for r in row..(row + row_span) {
        for c in col..(col + col_span) {
            if let Some(row_data) = occupied.get(r) {
                if row_data.get(c).copied().unwrap_or(false) {
                    return false;
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TrackSize Tests
    // =========================================================================

    #[test]
    fn test_track_size_default() {
        assert_eq!(TrackSize::default(), TrackSize::Fr(1.0));
    }

    #[test]
    fn test_track_size_px() {
        let size = TrackSize::px(100.0);
        assert_eq!(size, TrackSize::Px(100.0));
    }

    #[test]
    fn test_track_size_fr() {
        let size = TrackSize::fr(2.0);
        assert_eq!(size, TrackSize::Fr(2.0));
    }

    // =========================================================================
    // GridTemplate Tests
    // =========================================================================

    #[test]
    fn test_grid_template_new() {
        let template = GridTemplate::new();
        assert!(template.columns.is_empty());
        assert!(template.rows.is_empty());
        assert_eq!(template.column_gap, 0.0);
        assert_eq!(template.row_gap, 0.0);
    }

    #[test]
    fn test_grid_template_columns() {
        let template = GridTemplate::columns([TrackSize::px(100.0), TrackSize::fr(1.0)]);
        assert_eq!(template.columns.len(), 2);
        assert_eq!(template.columns[0], TrackSize::Px(100.0));
        assert_eq!(template.columns[1], TrackSize::Fr(1.0));
    }

    #[test]
    fn test_grid_template_twelve_column() {
        let template = GridTemplate::twelve_column();
        assert_eq!(template.columns.len(), 12);
        assert_eq!(template.column_gap, 16.0);
        assert_eq!(template.row_gap, 16.0);
    }

    #[test]
    fn test_grid_template_builder() {
        let template = GridTemplate::columns([TrackSize::fr(1.0), TrackSize::fr(2.0)])
            .with_rows([TrackSize::px(50.0)])
            .with_gap(8.0);

        assert_eq!(template.columns.len(), 2);
        assert_eq!(template.rows.len(), 1);
        assert_eq!(template.column_gap, 8.0);
        assert_eq!(template.row_gap, 8.0);
    }

    #[test]
    fn test_grid_template_with_area() {
        let template = GridTemplate::twelve_column()
            .with_area("header", GridArea::row_span(0, 0, 12))
            .with_area("sidebar", GridArea::col_span(0, 1, 4));

        assert!(template.areas.contains_key("header"));
        assert!(template.areas.contains_key("sidebar"));
    }

    // =========================================================================
    // GridArea Tests
    // =========================================================================

    #[test]
    fn test_grid_area_new() {
        let area = GridArea::new(1, 2, 3, 4);
        assert_eq!(area.row_start, 1);
        assert_eq!(area.col_start, 2);
        assert_eq!(area.row_end, 3);
        assert_eq!(area.col_end, 4);
    }

    #[test]
    fn test_grid_area_cell() {
        let area = GridArea::cell(2, 3);
        assert_eq!(area.row_start, 2);
        assert_eq!(area.row_end, 3);
        assert_eq!(area.col_start, 3);
        assert_eq!(area.col_end, 4);
    }

    #[test]
    fn test_grid_area_row_span() {
        let area = GridArea::row_span(0, 0, 6);
        assert_eq!(area.row_span_count(), 1);
        assert_eq!(area.col_span_count(), 6);
    }

    #[test]
    fn test_grid_area_col_span() {
        let area = GridArea::col_span(0, 0, 3);
        assert_eq!(area.row_span_count(), 3);
        assert_eq!(area.col_span_count(), 1);
    }

    // =========================================================================
    // GridItem Tests
    // =========================================================================

    #[test]
    fn test_grid_item_new() {
        let item = GridItem::new();
        assert_eq!(item.column_span, 1);
        assert_eq!(item.row_span, 1);
    }

    #[test]
    fn test_grid_item_builder() {
        let item = GridItem::new()
            .column(2)
            .row(1)
            .span_columns(3)
            .span_rows(2);

        assert_eq!(item.column_start, 2);
        assert_eq!(item.row_start, 1);
        assert_eq!(item.column_span, 3);
        assert_eq!(item.row_span, 2);
    }

    #[test]
    fn test_grid_item_effective_span() {
        let item1 = GridItem::new().span_columns(2);
        assert_eq!(item1.effective_column_span(), 2);

        let mut item2 = GridItem::new();
        item2.column_start = 1;
        item2.column_end = 4;
        assert_eq!(item2.effective_column_span(), 3);
    }

    #[test]
    fn test_grid_item_in_area() {
        let item = GridItem::new().in_area("sidebar");
        assert_eq!(item.area, Some("sidebar".to_string()));
    }

    // =========================================================================
    // GridAlign Tests
    // =========================================================================

    #[test]
    fn test_grid_align_default() {
        assert_eq!(GridAlign::default(), GridAlign::Center);
    }

    // =========================================================================
    // compute_track_sizes Tests
    // =========================================================================

    #[test]
    fn test_compute_track_sizes_fixed() {
        let tracks = vec![TrackSize::Px(100.0), TrackSize::Px(200.0)];
        let result = compute_track_sizes(&tracks, 400.0, 0.0, &[]);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (0.0, 100.0));
        assert_eq!(result[1], (100.0, 200.0));
    }

    #[test]
    fn test_compute_track_sizes_fr() {
        let tracks = vec![TrackSize::Fr(1.0), TrackSize::Fr(1.0)];
        let result = compute_track_sizes(&tracks, 200.0, 0.0, &[]);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (0.0, 100.0));
        assert_eq!(result[1], (100.0, 100.0));
    }

    #[test]
    fn test_compute_track_sizes_mixed() {
        let tracks = vec![TrackSize::Px(50.0), TrackSize::Fr(1.0), TrackSize::Fr(1.0)];
        let result = compute_track_sizes(&tracks, 250.0, 0.0, &[]);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], (0.0, 50.0));
        assert_eq!(result[1], (50.0, 100.0));
        assert_eq!(result[2], (150.0, 100.0));
    }

    #[test]
    fn test_compute_track_sizes_with_gap() {
        let tracks = vec![TrackSize::Fr(1.0), TrackSize::Fr(1.0)];
        let result = compute_track_sizes(&tracks, 210.0, 10.0, &[]);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (0.0, 100.0));
        assert_eq!(result[1], (110.0, 100.0));
    }

    #[test]
    fn test_compute_track_sizes_auto() {
        let tracks = vec![TrackSize::Auto, TrackSize::Fr(1.0)];
        let content_sizes = vec![80.0];
        let result = compute_track_sizes(&tracks, 200.0, 0.0, &content_sizes);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (0.0, 80.0));
        assert_eq!(result[1], (80.0, 120.0));
    }

    #[test]
    fn test_compute_track_sizes_empty() {
        let result = compute_track_sizes(&[], 200.0, 0.0, &[]);
        assert!(result.is_empty());
    }

    // =========================================================================
    // compute_grid_layout Tests
    // =========================================================================

    #[test]
    fn test_compute_grid_layout_basic() {
        let template = GridTemplate::columns([TrackSize::Fr(1.0), TrackSize::Fr(1.0)]);
        let layout = compute_grid_layout(&template, 200.0, 100.0, &[(50.0, 50.0), (50.0, 50.0)]);

        assert_eq!(layout.columns.len(), 2);
        assert_eq!(layout.width, 200.0);
    }

    #[test]
    fn test_compute_grid_layout_twelve_column() {
        let template = GridTemplate::twelve_column();
        let layout = compute_grid_layout(&template, 1200.0, 400.0, &[]);

        assert_eq!(layout.columns.len(), 12);
        // With 16px gap between 12 columns = 11 * 16 = 176px for gaps
        // Remaining: 1200 - 176 = 1024px for columns
        // Each column: 1024 / 12 ≈ 85.33px
    }

    #[test]
    fn test_grid_layout_area_bounds() {
        let template = GridTemplate::columns([TrackSize::px(100.0), TrackSize::px(100.0)]);
        let layout = compute_grid_layout(&template, 200.0, 100.0, &[(50.0, 50.0)]);

        let bounds = layout.area_bounds(&GridArea::cell(0, 0));
        assert!(bounds.is_some());
        let (x, y, w, _h) = bounds.unwrap();
        assert_eq!(x, 0.0);
        assert_eq!(y, 0.0);
        assert_eq!(w, 100.0);
    }

    #[test]
    fn test_grid_layout_area_bounds_span() {
        let template = GridTemplate::columns([
            TrackSize::px(100.0),
            TrackSize::px(100.0),
            TrackSize::px(100.0),
        ]);
        let layout = compute_grid_layout(&template, 300.0, 100.0, &[(50.0, 50.0)]);

        let bounds = layout.area_bounds(&GridArea::row_span(0, 0, 2));
        assert!(bounds.is_some());
        let (x, y, w, _h) = bounds.unwrap();
        assert_eq!(x, 0.0);
        assert_eq!(y, 0.0);
        assert_eq!(w, 200.0);
    }

    // =========================================================================
    // auto_place_items Tests
    // =========================================================================

    #[test]
    fn test_auto_place_items_simple() {
        let template = GridTemplate::columns([TrackSize::fr(1.0), TrackSize::fr(1.0)]);
        let items = vec![GridItem::new(), GridItem::new(), GridItem::new()];

        let placements = auto_place_items(&template, &items, GridAutoFlow::Row);

        assert_eq!(placements.len(), 3);
        assert_eq!(placements[0], (0, 0));
        assert_eq!(placements[1], (0, 1));
        assert_eq!(placements[2], (1, 0));
    }

    #[test]
    fn test_auto_place_items_with_span() {
        let template =
            GridTemplate::columns([TrackSize::fr(1.0), TrackSize::fr(1.0), TrackSize::fr(1.0)]);
        let items = vec![
            GridItem::new().span_columns(2),
            GridItem::new(),
            GridItem::new(),
        ];

        let placements = auto_place_items(&template, &items, GridAutoFlow::Row);

        assert_eq!(placements.len(), 3);
        assert_eq!(placements[0], (0, 0)); // Spans 2 columns
        assert_eq!(placements[1], (0, 2)); // Fits in remaining column
        assert_eq!(placements[2], (1, 0)); // Next row
    }

    #[test]
    fn test_auto_place_items_explicit() {
        let template = GridTemplate::columns([TrackSize::fr(1.0), TrackSize::fr(1.0)]);
        let items = vec![GridItem::new().column(2).row(2), GridItem::new()];

        let placements = auto_place_items(&template, &items, GridAutoFlow::Row);

        assert_eq!(placements[0], (1, 1)); // Explicit (converted to 0-indexed)
        assert_eq!(placements[1], (0, 0)); // Auto-placed
    }

    #[test]
    fn test_auto_place_items_named_area() {
        let template = GridTemplate::columns([TrackSize::fr(1.0), TrackSize::fr(1.0)])
            .with_area("main", GridArea::cell(1, 1));
        let items = vec![GridItem::new().in_area("main"), GridItem::new()];

        let placements = auto_place_items(&template, &items, GridAutoFlow::Row);

        assert_eq!(placements[0], (1, 1)); // Named area
        assert_eq!(placements[1], (0, 0)); // Auto-placed
    }

    #[test]
    fn test_auto_place_items_column_flow() {
        let template = GridTemplate::columns([TrackSize::fr(1.0), TrackSize::fr(1.0)]);
        let items = vec![GridItem::new(), GridItem::new(), GridItem::new()];

        let placements = auto_place_items(&template, &items, GridAutoFlow::Column);

        assert_eq!(placements.len(), 3);
        // Column flow fills down first
        assert_eq!(placements[0], (0, 0));
        assert_eq!(placements[1], (1, 0));
        assert_eq!(placements[2], (2, 0));
    }

    // =========================================================================
    // GridAutoFlow Tests
    // =========================================================================

    #[test]
    fn test_grid_auto_flow_default() {
        assert_eq!(GridAutoFlow::default(), GridAutoFlow::Row);
    }

    #[test]
    fn test_grid_auto_flow_all_variants() {
        assert_eq!(GridAutoFlow::Row, GridAutoFlow::Row);
        assert_eq!(GridAutoFlow::Column, GridAutoFlow::Column);
        assert_eq!(GridAutoFlow::RowDense, GridAutoFlow::RowDense);
        assert_eq!(GridAutoFlow::ColumnDense, GridAutoFlow::ColumnDense);
    }

    #[test]
    fn test_grid_auto_flow_clone() {
        let flow = GridAutoFlow::ColumnDense;
        let cloned = flow;
        assert_eq!(flow, cloned);
    }

    #[test]
    fn test_grid_auto_flow_debug() {
        let flow = GridAutoFlow::RowDense;
        let debug = format!("{:?}", flow);
        assert!(debug.contains("RowDense"));
    }

    // =========================================================================
    // TrackSize Additional Tests
    // =========================================================================

    #[test]
    fn test_track_size_auto_const() {
        assert_eq!(TrackSize::AUTO, TrackSize::Auto);
    }

    #[test]
    fn test_track_size_min_content() {
        let size = TrackSize::MinContent;
        assert_eq!(size, TrackSize::MinContent);
    }

    #[test]
    fn test_track_size_max_content() {
        let size = TrackSize::MaxContent;
        assert_eq!(size, TrackSize::MaxContent);
    }

    #[test]
    fn test_track_size_clone() {
        let size = TrackSize::Fr(2.5);
        let cloned = size;
        assert_eq!(size, cloned);
    }

    #[test]
    fn test_track_size_debug() {
        let size = TrackSize::Px(100.0);
        let debug = format!("{:?}", size);
        assert!(debug.contains("Px"));
        assert!(debug.contains("100"));
    }

    #[test]
    fn test_track_size_serialize() {
        let size = TrackSize::Fr(1.5);
        let json = serde_json::to_string(&size).unwrap();
        assert!(json.contains("Fr"));
    }

    #[test]
    fn test_track_size_deserialize() {
        let json = r#"{"Px":200.0}"#;
        let size: TrackSize = serde_json::from_str(json).unwrap();
        assert_eq!(size, TrackSize::Px(200.0));
    }

    // =========================================================================
    // GridTemplate Additional Tests
    // =========================================================================

    #[test]
    fn test_grid_template_with_column_gap() {
        let template = GridTemplate::new().with_column_gap(20.0);
        assert_eq!(template.column_gap, 20.0);
        assert_eq!(template.row_gap, 0.0);
    }

    #[test]
    fn test_grid_template_with_row_gap() {
        let template = GridTemplate::new().with_row_gap(15.0);
        assert_eq!(template.row_gap, 15.0);
        assert_eq!(template.column_gap, 0.0);
    }

    #[test]
    fn test_grid_template_column_count() {
        let template =
            GridTemplate::columns([TrackSize::fr(1.0), TrackSize::fr(1.0), TrackSize::fr(1.0)]);
        assert_eq!(template.column_count(), 3);
    }

    #[test]
    fn test_grid_template_row_count() {
        let template = GridTemplate::new().with_rows([TrackSize::px(50.0), TrackSize::px(100.0)]);
        assert_eq!(template.row_count(), 2);
    }

    #[test]
    fn test_grid_template_empty_counts() {
        let template = GridTemplate::new();
        assert_eq!(template.column_count(), 0);
        assert_eq!(template.row_count(), 0);
    }

    #[test]
    fn test_grid_template_default() {
        let template = GridTemplate::default();
        assert!(template.columns.is_empty());
        assert!(template.areas.is_empty());
    }

    #[test]
    fn test_grid_template_clone() {
        let template = GridTemplate::twelve_column().with_area("header", GridArea::cell(0, 0));
        let cloned = template;
        assert_eq!(cloned.columns.len(), 12);
        assert!(cloned.areas.contains_key("header"));
    }

    #[test]
    fn test_grid_template_serialize() {
        let template = GridTemplate::columns([TrackSize::px(100.0)]);
        let json = serde_json::to_string(&template).unwrap();
        assert!(json.contains("columns"));
    }

    // =========================================================================
    // GridArea Additional Tests
    // =========================================================================

    #[test]
    fn test_grid_area_span_counts_zero() {
        let area = GridArea::new(5, 5, 5, 5); // Same start and end
        assert_eq!(area.row_span_count(), 0);
        assert_eq!(area.col_span_count(), 0);
    }

    #[test]
    fn test_grid_area_span_counts_saturating() {
        let area = GridArea::new(10, 10, 5, 5); // End before start
        assert_eq!(area.row_span_count(), 0);
        assert_eq!(area.col_span_count(), 0);
    }

    #[test]
    fn test_grid_area_clone() {
        let area = GridArea::new(1, 2, 3, 4);
        let cloned = area;
        assert_eq!(area, cloned);
    }

    #[test]
    fn test_grid_area_debug() {
        let area = GridArea::cell(0, 0);
        let debug = format!("{:?}", area);
        assert!(debug.contains("GridArea"));
    }

    #[test]
    fn test_grid_area_serialize() {
        let area = GridArea::new(0, 0, 2, 4);
        let json = serde_json::to_string(&area).unwrap();
        assert!(json.contains("row_start"));
    }

    #[test]
    fn test_grid_area_deserialize() {
        let json = r#"{"row_start":1,"row_end":3,"col_start":0,"col_end":2}"#;
        let area: GridArea = serde_json::from_str(json).unwrap();
        assert_eq!(area.row_start, 1);
        assert_eq!(area.col_end, 2);
    }

    // =========================================================================
    // GridItem Additional Tests
    // =========================================================================

    #[test]
    fn test_grid_item_default() {
        let item = GridItem::default();
        assert_eq!(item.column_start, 0);
        assert_eq!(item.row_start, 0);
        assert!(item.area.is_none());
    }

    #[test]
    fn test_grid_item_justify_self() {
        let item = GridItem::new().justify_self(GridAlign::End);
        assert_eq!(item.justify_self, Some(GridAlign::End));
    }

    #[test]
    fn test_grid_item_align_self() {
        let item = GridItem::new().align_self(GridAlign::Stretch);
        assert_eq!(item.align_self, Some(GridAlign::Stretch));
    }

    #[test]
    fn test_grid_item_effective_row_span() {
        let item = GridItem::new().span_rows(3);
        assert_eq!(item.effective_row_span(), 3);

        let mut item2 = GridItem::new();
        item2.row_start = 1;
        item2.row_end = 5;
        assert_eq!(item2.effective_row_span(), 4);
    }

    #[test]
    fn test_grid_item_effective_span_minimum() {
        let mut item = GridItem::new();
        item.column_span = 0; // Edge case
        assert_eq!(item.effective_column_span(), 1); // Min 1

        item.row_span = 0;
        assert_eq!(item.effective_row_span(), 1); // Min 1
    }

    #[test]
    fn test_grid_item_clone() {
        let item = GridItem::new().column(2).row(3).span_columns(2);
        let cloned = item;
        assert_eq!(cloned.column_start, 2);
        assert_eq!(cloned.row_start, 3);
    }

    #[test]
    fn test_grid_item_serialize() {
        let item = GridItem::new().column(1).row(1);
        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("column_start"));
    }

    // =========================================================================
    // GridAlign Additional Tests
    // =========================================================================

    #[test]
    fn test_grid_align_all_variants() {
        assert_eq!(GridAlign::Start, GridAlign::Start);
        assert_eq!(GridAlign::End, GridAlign::End);
        assert_eq!(GridAlign::Center, GridAlign::Center);
        assert_eq!(GridAlign::Stretch, GridAlign::Stretch);
    }

    #[test]
    fn test_grid_align_clone() {
        let align = GridAlign::Stretch;
        let cloned = align;
        assert_eq!(align, cloned);
    }

    #[test]
    fn test_grid_align_debug() {
        let align = GridAlign::Start;
        let debug = format!("{:?}", align);
        assert!(debug.contains("Start"));
    }

    #[test]
    fn test_grid_align_serialize() {
        let align = GridAlign::End;
        let json = serde_json::to_string(&align).unwrap();
        assert!(json.contains("End"));
    }

    // =========================================================================
    // GridLayout Tests
    // =========================================================================

    #[test]
    fn test_grid_layout_default() {
        let layout = GridLayout::default();
        assert!(layout.columns.is_empty());
        assert!(layout.rows.is_empty());
        assert_eq!(layout.width, 0.0);
        assert_eq!(layout.height, 0.0);
    }

    #[test]
    fn test_grid_layout_area_bounds_out_of_range() {
        let layout = GridLayout {
            columns: vec![(0.0, 100.0)],
            rows: vec![(0.0, 50.0)],
            width: 100.0,
            height: 50.0,
        };

        // Out of range area
        let bounds = layout.area_bounds(&GridArea::cell(10, 10));
        assert!(bounds.is_none());
    }

    #[test]
    fn test_grid_layout_area_bounds_partial_out_of_range() {
        let layout = GridLayout {
            columns: vec![(0.0, 100.0), (100.0, 100.0)],
            rows: vec![(0.0, 50.0)],
            width: 200.0,
            height: 50.0,
        };

        // Area extends beyond grid - should clamp
        let bounds = layout.area_bounds(&GridArea::new(0, 0, 5, 5));
        assert!(bounds.is_some());
        let (_, _, w, _) = bounds.unwrap();
        assert_eq!(w, 200.0); // Clamped to available columns
    }

    #[test]
    fn test_grid_layout_item_bounds() {
        let layout = GridLayout {
            columns: vec![(0.0, 100.0), (100.0, 100.0)],
            rows: vec![(0.0, 50.0), (50.0, 50.0)],
            width: 200.0,
            height: 100.0,
        };

        let item = GridItem::new().span_columns(2);
        let bounds = layout.item_bounds(&item, 0, 0);
        assert!(bounds.is_some());
        let (x, y, w, h) = bounds.unwrap();
        assert_eq!(x, 0.0);
        assert_eq!(y, 0.0);
        assert_eq!(w, 200.0);
        assert_eq!(h, 50.0);
    }

    #[test]
    fn test_grid_layout_clone() {
        let layout = GridLayout {
            columns: vec![(0.0, 100.0)],
            rows: vec![(0.0, 50.0)],
            width: 100.0,
            height: 50.0,
        };
        let cloned = layout;
        assert_eq!(cloned.width, 100.0);
    }

    #[test]
    fn test_grid_layout_debug() {
        let layout = GridLayout::default();
        let debug = format!("{:?}", layout);
        assert!(debug.contains("GridLayout"));
    }

    // =========================================================================
    // compute_track_sizes Edge Cases
    // =========================================================================

    #[test]
    fn test_compute_track_sizes_all_fr() {
        let tracks = vec![TrackSize::Fr(1.0), TrackSize::Fr(2.0), TrackSize::Fr(1.0)];
        let result = compute_track_sizes(&tracks, 400.0, 0.0, &[]);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].1, 100.0); // 1/4
        assert_eq!(result[1].1, 200.0); // 2/4
        assert_eq!(result[2].1, 100.0); // 1/4
    }

    #[test]
    fn test_compute_track_sizes_zero_fr() {
        let tracks = vec![TrackSize::Fr(0.0), TrackSize::Fr(1.0)];
        let result = compute_track_sizes(&tracks, 200.0, 0.0, &[]);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].1, 0.0);
        assert_eq!(result[1].1, 200.0);
    }

    #[test]
    fn test_compute_track_sizes_insufficient_space() {
        let tracks = vec![TrackSize::Px(300.0), TrackSize::Fr(1.0)];
        let result = compute_track_sizes(&tracks, 200.0, 0.0, &[]);

        // Fixed takes precedence, fr gets 0
        assert_eq!(result[0].1, 300.0);
        assert_eq!(result[1].1, 0.0);
    }

    #[test]
    fn test_compute_track_sizes_large_gap() {
        let tracks = vec![TrackSize::Fr(1.0), TrackSize::Fr(1.0)];
        let result = compute_track_sizes(&tracks, 100.0, 200.0, &[]);

        // Gap is larger than available space - tracks get 0
        assert!(result[0].1 <= 0.0);
    }

    #[test]
    fn test_compute_track_sizes_min_content() {
        let tracks = vec![TrackSize::MinContent, TrackSize::Fr(1.0)];
        let content = vec![60.0, 0.0];
        let result = compute_track_sizes(&tracks, 200.0, 0.0, &content);

        assert_eq!(result[0].1, 60.0);
        assert_eq!(result[1].1, 140.0);
    }

    #[test]
    fn test_compute_track_sizes_max_content() {
        let tracks = vec![TrackSize::MaxContent, TrackSize::Fr(1.0)];
        let content = vec![80.0];
        let result = compute_track_sizes(&tracks, 200.0, 0.0, &content);

        assert_eq!(result[0].1, 80.0);
        assert_eq!(result[1].1, 120.0);
    }

    // =========================================================================
    // compute_grid_layout Edge Cases
    // =========================================================================

    #[test]
    fn test_compute_grid_layout_empty_template() {
        let template = GridTemplate::new();
        let layout = compute_grid_layout(&template, 200.0, 100.0, &[]);

        assert!(layout.columns.is_empty());
        assert_eq!(layout.width, 0.0);
    }

    #[test]
    fn test_compute_grid_layout_auto_rows() {
        let template = GridTemplate::columns([TrackSize::fr(1.0), TrackSize::fr(1.0)]);
        let children = vec![(50.0, 30.0), (50.0, 30.0), (50.0, 30.0), (50.0, 30.0)];
        let layout = compute_grid_layout(&template, 200.0, 200.0, &children);

        // 4 items in 2 columns = 2 rows
        assert_eq!(layout.rows.len(), 2);
    }

    #[test]
    fn test_compute_grid_layout_with_gaps() {
        let template = GridTemplate::columns([TrackSize::fr(1.0), TrackSize::fr(1.0)])
            .with_rows([TrackSize::px(50.0)])
            .with_gap(10.0);
        let layout = compute_grid_layout(&template, 210.0, 50.0, &[]);

        // Total width should account for gap
        let col1_end = layout.columns[0].0 + layout.columns[0].1;
        let col2_start = layout.columns[1].0;
        assert!((col2_start - col1_end - 10.0).abs() < 0.01);
    }

    // =========================================================================
    // auto_place_items Edge Cases
    // =========================================================================

    #[test]
    fn test_auto_place_items_empty() {
        let template = GridTemplate::columns([TrackSize::fr(1.0)]);
        let placements = auto_place_items(&template, &[], GridAutoFlow::Row);
        assert!(placements.is_empty());
    }

    #[test]
    fn test_auto_place_items_single_column() {
        let template = GridTemplate::columns([TrackSize::fr(1.0)]);
        let items = vec![GridItem::new(), GridItem::new(), GridItem::new()];
        let placements = auto_place_items(&template, &items, GridAutoFlow::Row);

        assert_eq!(placements[0], (0, 0));
        assert_eq!(placements[1], (1, 0));
        assert_eq!(placements[2], (2, 0));
    }

    #[test]
    fn test_auto_place_items_span_exceeds_grid() {
        let template = GridTemplate::columns([TrackSize::fr(1.0), TrackSize::fr(1.0)]);
        let items = vec![GridItem::new().span_columns(5)]; // Spans more than available
        let placements = auto_place_items(&template, &items, GridAutoFlow::Row);

        // Should still place somewhere
        assert_eq!(placements.len(), 1);
    }

    #[test]
    fn test_auto_place_items_row_span() {
        let template = GridTemplate::columns([TrackSize::fr(1.0), TrackSize::fr(1.0)]);
        let items = vec![
            GridItem::new().span_rows(2),
            GridItem::new(),
            GridItem::new(),
        ];
        let placements = auto_place_items(&template, &items, GridAutoFlow::Row);

        // First item spans 2 rows, second should go to column 1
        assert_eq!(placements[0], (0, 0));
        assert_eq!(placements[1], (0, 1));
    }

    #[test]
    fn test_auto_place_items_missing_area() {
        let template = GridTemplate::columns([TrackSize::fr(1.0)]);
        let items = vec![GridItem::new().in_area("nonexistent")];
        let placements = auto_place_items(&template, &items, GridAutoFlow::Row);

        // Should fall back to auto-placement
        assert_eq!(placements[0], (0, 0));
    }

    #[test]
    fn test_auto_place_items_column_dense() {
        let template = GridTemplate::columns([TrackSize::fr(1.0), TrackSize::fr(1.0)]);
        let items = vec![GridItem::new(), GridItem::new()];
        let placements = auto_place_items(&template, &items, GridAutoFlow::ColumnDense);

        // ColumnDense should fill columns first
        assert_eq!(placements.len(), 2);
    }
}
