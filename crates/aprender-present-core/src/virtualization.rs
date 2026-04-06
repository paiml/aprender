// Scroll Virtualization - WASM-first list/grid virtualization
//
// Provides:
// - Virtual scrolling for large lists
// - Only renders visible items + overscan
// - Variable item heights support
// - Grid virtualization
// - Infinite scroll support
// - Scroll position restoration

use std::collections::HashMap;
use std::ops::Range;

/// Index of an item in a virtualized list
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ItemIndex(pub usize);

impl ItemIndex {
    pub fn as_usize(self) -> usize {
        self.0
    }
}

impl From<usize> for ItemIndex {
    fn from(v: usize) -> Self {
        Self(v)
    }
}

/// Configuration for virtualized list
#[derive(Debug, Clone)]
pub struct VirtualListConfig {
    /// Estimated height of each item (used when actual height unknown)
    pub estimated_item_height: f32,
    /// Number of items to render above/below visible area
    pub overscan_count: usize,
    /// Enable variable height items
    pub variable_heights: bool,
    /// Initial scroll position
    pub initial_scroll: f32,
    /// Scroll threshold for triggering load more
    pub load_more_threshold: f32,
}

impl Default for VirtualListConfig {
    fn default() -> Self {
        Self {
            estimated_item_height: 50.0,
            overscan_count: 3,
            variable_heights: false,
            initial_scroll: 0.0,
            load_more_threshold: 100.0,
        }
    }
}

/// Visible range information
#[derive(Debug, Clone, PartialEq)]
pub struct VisibleRange {
    /// First visible item index
    pub start: usize,
    /// Last visible item index (exclusive)
    pub end: usize,
    /// First item to render (including overscan)
    pub render_start: usize,
    /// Last item to render (exclusive, including overscan)
    pub render_end: usize,
    /// Offset for the first rendered item
    pub offset: f32,
}

impl VisibleRange {
    /// Get range of visible items
    pub fn visible_range(&self) -> Range<usize> {
        self.start..self.end
    }

    /// Get range of items to render
    pub fn render_range(&self) -> Range<usize> {
        contract_pre_render!();
        self.render_start..self.render_end
    }

    /// Check if index is visible
    pub fn is_visible(&self, index: usize) -> bool {
        index >= self.start && index < self.end
    }

    /// Check if index should be rendered
    pub fn should_render(&self, index: usize) -> bool {
        contract_pre_render!();
        index >= self.render_start && index < self.render_end
    }

    /// Number of visible items
    pub fn visible_count(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Number of items to render
    pub fn render_count(&self) -> usize {
        contract_pre_render!();
        self.render_end.saturating_sub(self.render_start)
    }
}

/// Item layout information
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ItemLayout {
    /// Y position of the item
    pub y: f32,
    /// Height of the item
    pub height: f32,
}

impl ItemLayout {
    pub fn new(y: f32, height: f32) -> Self {
        Self { y, height }
    }

    /// Get the bottom edge of this item
    pub fn bottom(&self) -> f32 {
        self.y + self.height
    }
}

/// Virtualized list state
pub struct VirtualList {
    config: VirtualListConfig,
    /// Total number of items
    item_count: usize,
    /// Known item heights (for variable height lists)
    item_heights: HashMap<usize, f32>,
    /// Cached item positions
    item_positions: Vec<f32>,
    /// Whether positions need recalculation
    positions_dirty: bool,
    /// Current scroll position
    scroll_position: f32,
    /// Viewport height
    viewport_height: f32,
    /// Total content height
    content_height: f32,
    /// Currently visible range
    visible_range: Option<VisibleRange>,
}

impl Default for VirtualList {
    fn default() -> Self {
        Self::new(VirtualListConfig::default())
    }
}

impl VirtualList {
    pub fn new(config: VirtualListConfig) -> Self {
        let initial_scroll = config.initial_scroll;
        Self {
            config,
            item_count: 0,
            item_heights: HashMap::new(),
            item_positions: Vec::new(),
            positions_dirty: true,
            scroll_position: initial_scroll,
            viewport_height: 0.0,
            content_height: 0.0,
            visible_range: None,
        }
    }

    /// Set total item count
    pub fn set_item_count(&mut self, count: usize) {
        if count != self.item_count {
            self.item_count = count;
            self.positions_dirty = true;
        }
    }

    /// Get total item count
    pub fn item_count(&self) -> usize {
        self.item_count
    }

    /// Set viewport height
    pub fn set_viewport_height(&mut self, height: f32) {
        if (height - self.viewport_height).abs() > 0.1 {
            self.viewport_height = height;
            self.update_visible_range();
        }
    }

    /// Get viewport height
    pub fn viewport_height(&self) -> f32 {
        self.viewport_height
    }

    /// Set scroll position
    pub fn set_scroll_position(&mut self, position: f32) {
        let clamped = position.max(0.0).min(self.max_scroll());
        if (clamped - self.scroll_position).abs() > 0.1 {
            self.scroll_position = clamped;
            self.update_visible_range();
        }
    }

    /// Get current scroll position
    pub fn scroll_position(&self) -> f32 {
        self.scroll_position
    }

    /// Get maximum scroll position
    pub fn max_scroll(&self) -> f32 {
        (self.calculate_content_height() - self.viewport_height).max(0.0)
    }

    /// Calculate content height without caching
    fn calculate_content_height(&self) -> f32 {
        if !self.config.variable_heights {
            return self.item_count as f32 * self.config.estimated_item_height;
        }

        let mut height = 0.0;
        for i in 0..self.item_count {
            height += self.get_item_height(i);
        }
        height
    }

    /// Scroll by delta
    pub fn scroll_by(&mut self, delta: f32) {
        self.set_scroll_position(self.scroll_position + delta);
    }

    /// Scroll to specific item
    pub fn scroll_to_item(&mut self, index: usize, align: ScrollAlign) {
        if index >= self.item_count {
            return;
        }

        if self.positions_dirty {
            self.recalculate_positions();
        }
        let item_y = self.get_item_position(index);
        let item_height = self.get_item_height(index);

        let new_scroll = match align {
            ScrollAlign::Start => item_y,
            ScrollAlign::Center => item_y - (self.viewport_height - item_height) / 2.0,
            ScrollAlign::End => item_y - self.viewport_height + item_height,
            ScrollAlign::Auto => {
                // Only scroll if item is not fully visible
                if item_y < self.scroll_position {
                    item_y
                } else if item_y + item_height > self.scroll_position + self.viewport_height {
                    item_y + item_height - self.viewport_height
                } else {
                    self.scroll_position
                }
            }
        };

        self.set_scroll_position(new_scroll);
    }

    /// Set height for a specific item
    pub fn set_item_height(&mut self, index: usize, height: f32) {
        if self.config.variable_heights {
            self.item_heights.insert(index, height);
            self.positions_dirty = true;
        }
    }

    /// Get height for a specific item
    pub fn get_item_height(&self, index: usize) -> f32 {
        if self.config.variable_heights {
            self.item_heights
                .get(&index)
                .copied()
                .unwrap_or(self.config.estimated_item_height)
        } else {
            self.config.estimated_item_height
        }
    }

    /// Get position for a specific item
    pub fn get_item_position(&self, index: usize) -> f32 {
        if index == 0 {
            return 0.0;
        }

        // For fixed height, calculate directly
        if !self.config.variable_heights {
            return index as f32 * self.config.estimated_item_height;
        }

        // For variable heights, use cached positions if available
        if index < self.item_positions.len() {
            self.item_positions[index]
        } else {
            // Calculate position on the fly
            let mut y = 0.0;
            for i in 0..index {
                y += self.get_item_height(i);
            }
            y
        }
    }

    /// Get layout for a specific item
    pub fn get_item_layout(&self, index: usize) -> ItemLayout {
        ItemLayout {
            y: self.get_item_position(index),
            height: self.get_item_height(index),
        }
    }

    /// Get total content height
    pub fn content_height(&self) -> f32 {
        self.calculate_content_height()
    }

    /// Get currently visible range
    pub fn visible_range(&self) -> Option<&VisibleRange> {
        self.visible_range.as_ref()
    }

    /// Check if we're near the end (for infinite scroll)
    pub fn is_near_end(&self) -> bool {
        self.scroll_position + self.viewport_height + self.config.load_more_threshold
            >= self.content_height
    }

    /// Check if we're near the start
    pub fn is_near_start(&self) -> bool {
        self.scroll_position <= self.config.load_more_threshold
    }

    /// Recalculate visible range
    fn update_visible_range(&mut self) {
        if self.positions_dirty {
            self.recalculate_positions();
        }

        if self.item_count == 0 || self.viewport_height <= 0.0 {
            self.visible_range = None;
            return;
        }

        // Find first visible item
        let start = self.find_item_at_position(self.scroll_position);
        let end = self.find_item_at_position(self.scroll_position + self.viewport_height) + 1;
        let end = end.min(self.item_count);

        // Calculate render range with overscan
        let render_start = start.saturating_sub(self.config.overscan_count);
        let render_end = (end + self.config.overscan_count).min(self.item_count);

        // Calculate offset for first rendered item
        let offset = self.get_item_position(render_start);

        self.visible_range = Some(VisibleRange {
            start,
            end,
            render_start,
            render_end,
            offset,
        });
    }

    /// Find item at a given scroll position
    fn find_item_at_position(&self, position: f32) -> usize {
        if position <= 0.0 {
            return 0;
        }

        if !self.config.variable_heights {
            // Fast path for fixed height
            return (position / self.config.estimated_item_height) as usize;
        }

        // Binary search for variable heights
        let mut low = 0;
        let mut high = self.item_count;

        while low < high {
            let mid = (low + high) / 2;
            let item_pos = self.get_item_position(mid);

            if item_pos <= position {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        low.saturating_sub(1)
    }

    /// Recalculate all positions
    fn recalculate_positions(&mut self) {
        self.item_positions.clear();
        self.item_positions.reserve(self.item_count);

        let mut current_y = 0.0;
        for i in 0..self.item_count {
            self.item_positions.push(current_y);
            current_y += self.get_item_height(i);
        }

        self.content_height = current_y;
        self.positions_dirty = false;
    }

    /// Reset scroll position
    pub fn reset(&mut self) {
        self.scroll_position = 0.0;
        self.visible_range = None;
        self.update_visible_range();
    }
}

/// Scroll alignment options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollAlign {
    /// Align item to start of viewport
    Start,
    /// Align item to center of viewport
    Center,
    /// Align item to end of viewport
    End,
    /// Only scroll if item not visible
    Auto,
}

/// Grid cell position
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridCell {
    pub row: usize,
    pub col: usize,
}

impl GridCell {
    pub fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }
}

/// Configuration for virtualized grid
#[derive(Debug, Clone)]
pub struct VirtualGridConfig {
    /// Number of columns
    pub columns: usize,
    /// Cell width
    pub cell_width: f32,
    /// Cell height
    pub cell_height: f32,
    /// Gap between cells
    pub gap: f32,
    /// Number of rows to render above/below visible area
    pub overscan_rows: usize,
}

impl Default for VirtualGridConfig {
    fn default() -> Self {
        Self {
            columns: 3,
            cell_width: 100.0,
            cell_height: 100.0,
            gap: 8.0,
            overscan_rows: 2,
        }
    }
}

/// Visible grid range
#[derive(Debug, Clone, PartialEq)]
pub struct VisibleGridRange {
    /// First visible row
    pub start_row: usize,
    /// Last visible row (exclusive)
    pub end_row: usize,
    /// First row to render (including overscan)
    pub render_start_row: usize,
    /// Last row to render (exclusive, including overscan)
    pub render_end_row: usize,
    /// Number of columns
    pub columns: usize,
    /// Y offset for first rendered row
    pub offset: f32,
}

impl VisibleGridRange {
    /// Get all cells that should be rendered
    pub fn cells_to_render(&self, total_items: usize) -> Vec<GridCell> {
        contract_pre_render!();
        let mut cells = Vec::new();
        for row in self.render_start_row..self.render_end_row {
            for col in 0..self.columns {
                let index = row * self.columns + col;
                if index < total_items {
                    cells.push(GridCell::new(row, col));
                }
            }
        }
        cells
    }

    /// Check if a cell should be rendered
    pub fn should_render_cell(&self, row: usize, col: usize) -> bool {
        contract_pre_render!();
        row >= self.render_start_row && row < self.render_end_row && col < self.columns
    }
}

/// Cell layout information
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellLayout {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl CellLayout {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

/// Virtualized grid state
pub struct VirtualGrid {
    config: VirtualGridConfig,
    /// Total number of items
    item_count: usize,
    /// Current scroll position
    scroll_position: f32,
    /// Viewport height
    viewport_height: f32,
    /// Currently visible range
    visible_range: Option<VisibleGridRange>,
}

impl Default for VirtualGrid {
    fn default() -> Self {
        Self::new(VirtualGridConfig::default())
    }
}

impl VirtualGrid {
    pub fn new(config: VirtualGridConfig) -> Self {
        Self {
            config,
            item_count: 0,
            scroll_position: 0.0,
            viewport_height: 0.0,
            visible_range: None,
        }
    }

    /// Set total item count
    pub fn set_item_count(&mut self, count: usize) {
        if count != self.item_count {
            self.item_count = count;
            self.update_visible_range();
        }
    }

    /// Get total item count
    pub fn item_count(&self) -> usize {
        self.item_count
    }

    /// Get row count
    pub fn row_count(&self) -> usize {
        self.item_count.div_ceil(self.config.columns)
    }

    /// Set viewport height
    pub fn set_viewport_height(&mut self, height: f32) {
        if (height - self.viewport_height).abs() > 0.1 {
            self.viewport_height = height;
            self.update_visible_range();
        }
    }

    /// Get viewport height
    pub fn viewport_height(&self) -> f32 {
        self.viewport_height
    }

    /// Set scroll position
    pub fn set_scroll_position(&mut self, position: f32) {
        let clamped = position.max(0.0).min(self.max_scroll());
        if (clamped - self.scroll_position).abs() > 0.1 {
            self.scroll_position = clamped;
            self.update_visible_range();
        }
    }

    /// Get current scroll position
    pub fn scroll_position(&self) -> f32 {
        self.scroll_position
    }

    /// Get maximum scroll position
    pub fn max_scroll(&self) -> f32 {
        (self.content_height() - self.viewport_height).max(0.0)
    }

    /// Scroll by delta
    pub fn scroll_by(&mut self, delta: f32) {
        self.set_scroll_position(self.scroll_position + delta);
    }

    /// Scroll to specific item
    pub fn scroll_to_item(&mut self, index: usize, align: ScrollAlign) {
        if index >= self.item_count {
            return;
        }

        let row = index / self.config.columns;
        let row_y = self.row_position(row);

        let new_scroll = match align {
            ScrollAlign::Start => row_y,
            ScrollAlign::Center => row_y - (self.viewport_height - self.row_height()) / 2.0,
            ScrollAlign::End => row_y - self.viewport_height + self.row_height(),
            ScrollAlign::Auto => {
                if row_y < self.scroll_position {
                    row_y
                } else if row_y + self.row_height() > self.scroll_position + self.viewport_height {
                    row_y + self.row_height() - self.viewport_height
                } else {
                    self.scroll_position
                }
            }
        };

        self.set_scroll_position(new_scroll);
    }

    /// Get row height (cell height + gap)
    pub fn row_height(&self) -> f32 {
        self.config.cell_height + self.config.gap
    }

    /// Get row position
    pub fn row_position(&self, row: usize) -> f32 {
        row as f32 * self.row_height()
    }

    /// Get content height
    pub fn content_height(&self) -> f32 {
        let rows = self.row_count();
        if rows == 0 {
            0.0
        } else {
            (rows as f32).mul_add(self.config.cell_height, (rows - 1) as f32 * self.config.gap)
        }
    }

    /// Get cell layout by index
    pub fn get_cell_layout(&self, index: usize) -> CellLayout {
        let row = index / self.config.columns;
        let col = index % self.config.columns;
        self.get_cell_layout_by_position(row, col)
    }

    /// Get cell layout by row/column
    pub fn get_cell_layout_by_position(&self, row: usize, col: usize) -> CellLayout {
        let x = col as f32 * (self.config.cell_width + self.config.gap);
        let y = row as f32 * (self.config.cell_height + self.config.gap);
        CellLayout::new(x, y, self.config.cell_width, self.config.cell_height)
    }

    /// Convert grid cell to item index
    pub fn cell_to_index(&self, cell: &GridCell) -> usize {
        cell.row * self.config.columns + cell.col
    }

    /// Convert item index to grid cell
    pub fn index_to_cell(&self, index: usize) -> GridCell {
        GridCell {
            row: index / self.config.columns,
            col: index % self.config.columns,
        }
    }

    /// Get visible range
    pub fn visible_range(&self) -> Option<&VisibleGridRange> {
        self.visible_range.as_ref()
    }

    /// Update visible range
    fn update_visible_range(&mut self) {
        if self.item_count == 0 || self.viewport_height <= 0.0 {
            self.visible_range = None;
            return;
        }

        let row_height = self.row_height();
        let start_row = (self.scroll_position / row_height) as usize;
        let visible_rows = (self.viewport_height / row_height).ceil() as usize + 1;
        let end_row = (start_row + visible_rows).min(self.row_count());

        let render_start_row = start_row.saturating_sub(self.config.overscan_rows);
        let render_end_row = (end_row + self.config.overscan_rows).min(self.row_count());

        let offset = render_start_row as f32 * row_height;

        self.visible_range = Some(VisibleGridRange {
            start_row,
            end_row,
            render_start_row,
            render_end_row,
            columns: self.config.columns,
            offset,
        });
    }

    /// Reset scroll position
    pub fn reset(&mut self) {
        self.scroll_position = 0.0;
        self.visible_range = None;
        self.update_visible_range();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn test_virtual_list_default() {
        let list = VirtualList::default();
        assert_eq!(list.item_count(), 0);
        assert_eq!(list.scroll_position(), 0.0);
    }

    #[test]
    fn test_virtual_list_set_item_count() {
        let mut list = VirtualList::default();
        list.set_item_count(100);
        assert_eq!(list.item_count(), 100);
    }

    #[test]
    fn test_virtual_list_viewport() {
        let mut list = VirtualList::default();
        list.set_viewport_height(500.0);
        assert_eq!(list.viewport_height(), 500.0);
    }

    #[test]
    fn test_virtual_list_scroll_position() {
        let mut list = VirtualList::default();
        list.set_item_count(100);
        list.set_viewport_height(500.0);

        list.set_scroll_position(100.0);
        assert_eq!(list.scroll_position(), 100.0);
    }

    #[test]
    fn test_virtual_list_scroll_clamped() {
        let mut list = VirtualList::default();
        list.set_item_count(10);
        list.set_viewport_height(500.0);

        // Should be clamped to 0
        list.set_scroll_position(-100.0);
        assert_eq!(list.scroll_position(), 0.0);
    }

    #[test]
    fn test_virtual_list_scroll_by() {
        let mut list = VirtualList::default();
        list.set_item_count(100);
        list.set_viewport_height(500.0);

        list.scroll_by(50.0);
        assert_eq!(list.scroll_position(), 50.0);

        list.scroll_by(25.0);
        assert_eq!(list.scroll_position(), 75.0);
    }

    #[test]
    fn test_virtual_list_content_height() {
        let config = VirtualListConfig {
            estimated_item_height: 40.0,
            ..Default::default()
        };
        let mut list = VirtualList::new(config);
        list.set_item_count(10);

        assert_eq!(list.content_height(), 400.0);
    }

    #[test]
    fn test_virtual_list_max_scroll() {
        let config = VirtualListConfig {
            estimated_item_height: 50.0,
            ..Default::default()
        };
        let mut list = VirtualList::new(config);
        list.set_item_count(20);
        list.set_viewport_height(400.0);

        // 20 items * 50 height = 1000, minus viewport 400 = 600 max scroll
        assert_eq!(list.max_scroll(), 600.0);
    }

    #[test]
    fn test_virtual_list_visible_range() {
        let config = VirtualListConfig {
            estimated_item_height: 50.0,
            overscan_count: 2,
            ..Default::default()
        };
        let mut list = VirtualList::new(config);
        list.set_item_count(100);
        list.set_viewport_height(200.0);

        let range = list.visible_range().unwrap();
        // 200 / 50 = 4 visible items (0-3), plus item 4 starts at position 200
        // so items 0-4 are at least partially visible
        assert_eq!(range.start, 0);
        assert_eq!(range.end, 5);
        assert_eq!(range.render_start, 0);
        assert_eq!(range.render_end, 7); // 5 + 2 overscan
    }

    #[test]
    fn test_virtual_list_visible_range_scrolled() {
        let config = VirtualListConfig {
            estimated_item_height: 50.0,
            overscan_count: 2,
            ..Default::default()
        };
        let mut list = VirtualList::new(config);
        list.set_item_count(100);
        list.set_viewport_height(200.0);
        list.set_scroll_position(250.0);

        let range = list.visible_range().unwrap();
        // 250 / 50 = 5, (250 + 200) / 50 = 9, + 1 = 10
        assert_eq!(range.start, 5);
        assert_eq!(range.end, 10);
        assert_eq!(range.render_start, 3); // 5 - 2
        assert_eq!(range.render_end, 12); // 10 + 2
    }

    #[test]
    fn test_virtual_list_scroll_to_item_start() {
        let config = VirtualListConfig {
            estimated_item_height: 50.0,
            ..Default::default()
        };
        let mut list = VirtualList::new(config);
        list.set_item_count(100);
        list.set_viewport_height(200.0);

        list.scroll_to_item(10, ScrollAlign::Start);
        assert_eq!(list.scroll_position(), 500.0);
    }

    #[test]
    fn test_virtual_list_scroll_to_item_center() {
        let config = VirtualListConfig {
            estimated_item_height: 50.0,
            ..Default::default()
        };
        let mut list = VirtualList::new(config);
        list.set_item_count(100);
        list.set_viewport_height(200.0);

        list.scroll_to_item(10, ScrollAlign::Center);
        // Item 10 at y=500, viewport=200, item height=50
        // center = 500 - (200 - 50) / 2 = 500 - 75 = 425
        assert_eq!(list.scroll_position(), 425.0);
    }

    #[test]
    fn test_virtual_list_scroll_to_item_end() {
        let config = VirtualListConfig {
            estimated_item_height: 50.0,
            ..Default::default()
        };
        let mut list = VirtualList::new(config);
        list.set_item_count(100);
        list.set_viewport_height(200.0);

        list.scroll_to_item(10, ScrollAlign::End);
        // Item 10 at y=500, viewport=200, item height=50
        // end = 500 - 200 + 50 = 350
        assert_eq!(list.scroll_position(), 350.0);
    }

    #[test]
    fn test_virtual_list_scroll_to_item_auto() {
        let config = VirtualListConfig {
            estimated_item_height: 50.0,
            ..Default::default()
        };
        let mut list = VirtualList::new(config);
        list.set_item_count(100);
        list.set_viewport_height(200.0);

        // Item 2 (at y=100) is already visible - shouldn't scroll
        list.scroll_to_item(2, ScrollAlign::Auto);
        assert_eq!(list.scroll_position(), 0.0);

        // Item 10 (at y=500) is not visible - should scroll
        list.scroll_to_item(10, ScrollAlign::Auto);
        assert!(list.scroll_position() > 0.0);
    }

    #[test]
    fn test_virtual_list_variable_heights() {
        let config = VirtualListConfig {
            estimated_item_height: 50.0,
            variable_heights: true,
            ..Default::default()
        };
        let mut list = VirtualList::new(config);
        list.set_item_count(10);

        list.set_item_height(2, 100.0);
        assert_eq!(list.get_item_height(2), 100.0);
        assert_eq!(list.get_item_height(3), 50.0); // Default
    }

    #[test]
    fn test_virtual_list_item_layout() {
        let config = VirtualListConfig {
            estimated_item_height: 50.0,
            ..Default::default()
        };
        let mut list = VirtualList::new(config);
        list.set_item_count(10);

        let layout = list.get_item_layout(5);
        assert_eq!(layout.y, 250.0);
        assert_eq!(layout.height, 50.0);
    }

    #[test]
    fn test_virtual_list_is_near_end() {
        let config = VirtualListConfig {
            estimated_item_height: 50.0,
            load_more_threshold: 100.0,
            ..Default::default()
        };
        let mut list = VirtualList::new(config);
        list.set_item_count(20); // 1000 total height
        list.set_viewport_height(300.0);

        assert!(!list.is_near_end());

        list.set_scroll_position(600.0); // Near end
        assert!(list.is_near_end());
    }

    #[test]
    fn test_virtual_list_is_near_start() {
        let config = VirtualListConfig {
            load_more_threshold: 100.0,
            ..Default::default()
        };
        let mut list = VirtualList::new(config);
        list.set_item_count(100);
        list.set_viewport_height(300.0);

        assert!(list.is_near_start());

        list.set_scroll_position(200.0);
        assert!(!list.is_near_start());
    }

    #[test]
    fn test_virtual_list_reset() {
        let mut list = VirtualList::default();
        list.set_item_count(100);
        list.set_viewport_height(300.0);
        list.set_scroll_position(500.0);

        list.reset();
        assert_eq!(list.scroll_position(), 0.0);
    }

    #[test]
    fn test_visible_range_methods() {
        let range = VisibleRange {
            start: 5,
            end: 10,
            render_start: 3,
            render_end: 12,
            offset: 150.0,
        };

        assert_eq!(range.visible_range(), 5..10);
        assert_eq!(range.render_range(), 3..12);
        assert_eq!(range.visible_count(), 5);
        assert_eq!(range.render_count(), 9);
        assert!(range.is_visible(7));
        assert!(!range.is_visible(2));
        assert!(range.should_render(5));
        assert!(!range.should_render(15));
    }

    #[test]
    fn test_item_layout() {
        let layout = ItemLayout::new(100.0, 50.0);
        assert_eq!(layout.y, 100.0);
        assert_eq!(layout.height, 50.0);
        assert_eq!(layout.bottom(), 150.0);
    }

    #[test]
    fn test_item_index() {
        let index = ItemIndex(42);
        assert_eq!(index.as_usize(), 42);

        let from_usize: ItemIndex = 100.into();
        assert_eq!(from_usize.0, 100);
    }

    // ========== Virtual Grid Tests ==========

    #[test]
    fn test_virtual_grid_default() {
        let grid = VirtualGrid::default();
        assert_eq!(grid.item_count(), 0);
        assert_eq!(grid.scroll_position(), 0.0);
    }

    #[test]
    fn test_virtual_grid_set_item_count() {
        let mut grid = VirtualGrid::default();
        grid.set_item_count(100);
        assert_eq!(grid.item_count(), 100);
    }

    #[test]
    fn test_virtual_grid_row_count() {
        let config = VirtualGridConfig {
            columns: 3,
            ..Default::default()
        };
        let mut grid = VirtualGrid::new(config);
        grid.set_item_count(10);
        assert_eq!(grid.row_count(), 4); // ceil(10/3) = 4
    }

    #[test]
    fn test_virtual_grid_viewport() {
        let mut grid = VirtualGrid::default();
        grid.set_viewport_height(500.0);
        assert_eq!(grid.viewport_height(), 500.0);
    }

    #[test]
    fn test_virtual_grid_scroll_position() {
        let mut grid = VirtualGrid::default();
        grid.set_item_count(100);
        grid.set_viewport_height(500.0);

        grid.set_scroll_position(200.0);
        assert_eq!(grid.scroll_position(), 200.0);
    }

    #[test]
    fn test_virtual_grid_content_height() {
        let config = VirtualGridConfig {
            columns: 3,
            cell_height: 100.0,
            gap: 10.0,
            ..Default::default()
        };
        let mut grid = VirtualGrid::new(config);
        grid.set_item_count(9); // 3 rows

        // 3 rows * 100 height + 2 gaps * 10 = 320
        assert_eq!(grid.content_height(), 320.0);
    }

    #[test]
    fn test_virtual_grid_cell_layout() {
        let config = VirtualGridConfig {
            columns: 3,
            cell_width: 100.0,
            cell_height: 80.0,
            gap: 10.0,
            ..Default::default()
        };
        let grid = VirtualGrid::new(config);

        // Item 0 at (0, 0)
        let layout = grid.get_cell_layout(0);
        assert_eq!(layout.x, 0.0);
        assert_eq!(layout.y, 0.0);

        // Item 1 at (110, 0)
        let layout = grid.get_cell_layout(1);
        assert_eq!(layout.x, 110.0);
        assert_eq!(layout.y, 0.0);

        // Item 3 at (0, 90) - second row
        let layout = grid.get_cell_layout(3);
        assert_eq!(layout.x, 0.0);
        assert_eq!(layout.y, 90.0);
    }

    #[test]
    fn test_virtual_grid_cell_conversion() {
        let config = VirtualGridConfig {
            columns: 4,
            ..Default::default()
        };
        let grid = VirtualGrid::new(config);

        let cell = grid.index_to_cell(10);
        assert_eq!(cell.row, 2);
        assert_eq!(cell.col, 2);

        assert_eq!(grid.cell_to_index(&cell), 10);
    }

    #[test]
    fn test_virtual_grid_visible_range() {
        let config = VirtualGridConfig {
            columns: 3,
            cell_height: 100.0,
            gap: 10.0,
            overscan_rows: 1,
            ..Default::default()
        };
        let mut grid = VirtualGrid::new(config);
        grid.set_item_count(30); // 10 rows
        grid.set_viewport_height(250.0);

        let range = grid.visible_range().unwrap();
        // Row height = 110, viewport = 250
        // Visible rows = ceil(250/110) + 1 = 4
        assert_eq!(range.start_row, 0);
        assert!(range.end_row >= 2);
    }

    #[test]
    fn test_virtual_grid_scroll_to_item() {
        let config = VirtualGridConfig {
            columns: 3,
            cell_height: 100.0,
            gap: 10.0,
            ..Default::default()
        };
        let mut grid = VirtualGrid::new(config);
        grid.set_item_count(30);
        grid.set_viewport_height(250.0);

        grid.scroll_to_item(15, ScrollAlign::Start); // Row 5
        assert_eq!(grid.scroll_position(), 550.0); // 5 * 110
    }

    #[test]
    fn test_virtual_grid_reset() {
        let mut grid = VirtualGrid::default();
        grid.set_item_count(100);
        grid.set_viewport_height(300.0);
        grid.set_scroll_position(500.0);

        grid.reset();
        assert_eq!(grid.scroll_position(), 0.0);
    }

    #[test]
    fn test_grid_cell() {
        let cell = GridCell::new(5, 2);
        assert_eq!(cell.row, 5);
        assert_eq!(cell.col, 2);
    }

    #[test]
    fn test_visible_grid_range_cells() {
        let range = VisibleGridRange {
            start_row: 2,
            end_row: 5,
            render_start_row: 1,
            render_end_row: 6,
            columns: 3,
            offset: 100.0,
        };

        // 5 render rows * 3 columns = 15 cells max
        let cells = range.cells_to_render(100);
        assert_eq!(cells.len(), 15);

        // With 10 items total (rows 0-3, plus 1 item in row 3):
        // Render rows 1-5, so items from row 1 onwards = items 3,4,5,6,7,8,9 = 7 items
        let cells = range.cells_to_render(10);
        assert_eq!(cells.len(), 7);
    }

    #[test]
    fn test_visible_grid_range_should_render() {
        let range = VisibleGridRange {
            start_row: 2,
            end_row: 5,
            render_start_row: 1,
            render_end_row: 6,
            columns: 3,
            offset: 100.0,
        };

        assert!(range.should_render_cell(3, 1));
        assert!(!range.should_render_cell(0, 0));
        assert!(!range.should_render_cell(3, 5)); // col out of range
    }

    #[test]
    fn test_cell_layout() {
        let layout = CellLayout::new(100.0, 200.0, 50.0, 60.0);
        assert_eq!(layout.x, 100.0);
        assert_eq!(layout.y, 200.0);
        assert_eq!(layout.width, 50.0);
        assert_eq!(layout.height, 60.0);
    }

    #[test]
    fn test_scroll_align_variants() {
        // Just make sure all variants exist and are comparable
        assert_ne!(ScrollAlign::Start, ScrollAlign::End);
        assert_ne!(ScrollAlign::Center, ScrollAlign::Auto);
    }

    #[test]
    fn test_virtual_list_empty() {
        let mut list = VirtualList::default();
        list.set_viewport_height(300.0);
        // Empty list should have no visible range
        assert!(list.visible_range().is_none());
    }

    #[test]
    fn test_virtual_grid_empty() {
        let mut grid = VirtualGrid::default();
        grid.set_viewport_height(300.0);
        // Empty grid should have no visible range
        assert!(grid.visible_range().is_none());
    }

    #[test]
    fn test_virtual_list_config_default() {
        let config = VirtualListConfig::default();
        assert_eq!(config.estimated_item_height, 50.0);
        assert_eq!(config.overscan_count, 3);
        assert!(!config.variable_heights);
        assert_eq!(config.initial_scroll, 0.0);
        assert_eq!(config.load_more_threshold, 100.0);
    }

    #[test]
    fn test_virtual_grid_config_default() {
        let config = VirtualGridConfig::default();
        assert_eq!(config.columns, 3);
        assert_eq!(config.cell_width, 100.0);
        assert_eq!(config.cell_height, 100.0);
        assert_eq!(config.gap, 8.0);
        assert_eq!(config.overscan_rows, 2);
    }
}
