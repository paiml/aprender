//! Scroll state management for TUI widgets
//!
//! Provides bounded scroll handling with page-based navigation.

/// Scroll state for navigating large datasets
///
/// Manages scroll position with bounds checking and page navigation.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScrollState {
    /// Current scroll offset (first visible row)
    offset: usize,
    /// Total number of rows
    total_rows: usize,
    /// Number of visible rows in viewport
    visible_rows: usize,
    /// Currently selected row (relative to data, not viewport)
    selected: Option<usize>,
}

impl ScrollState {
    /// Create a new scroll state
    ///
    /// # Arguments
    /// * `total_rows` - Total number of rows in the dataset
    /// * `visible_rows` - Number of rows visible in the viewport
    pub fn new(total_rows: usize, visible_rows: usize) -> Self {
        Self {
            offset: 0,
            total_rows,
            visible_rows,
            selected: None,
        }
    }

    /// Get current scroll offset
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Set scroll offset with bounds clamping
    pub fn set_offset(&mut self, offset: usize) {
        self.offset = self.clamp_offset(offset);
    }

    /// Get total row count
    #[inline]
    pub fn total_rows(&self) -> usize {
        self.total_rows
    }

    /// Update total row count
    pub fn set_total_rows(&mut self, total: usize) {
        self.total_rows = total;
        // Re-clamp offset if needed
        self.offset = self.clamp_offset(self.offset);
        // Re-clamp selection if needed
        if let Some(sel) = self.selected {
            if sel >= total {
                self.selected = if total > 0 { Some(total - 1) } else { None };
            }
        }
    }

    /// Get visible row count
    #[inline]
    pub fn visible_rows(&self) -> usize {
        self.visible_rows
    }

    /// Update visible row count
    pub fn set_visible_rows(&mut self, visible: usize) {
        self.visible_rows = visible;
        // Re-clamp offset if needed
        self.offset = self.clamp_offset(self.offset);
    }

    /// Get currently selected row
    #[inline]
    pub fn selected(&self) -> Option<usize> {
        self.selected
    }

    /// Set selected row with bounds checking
    pub fn set_selected(&mut self, row: Option<usize>) {
        self.selected = match row {
            Some(r) if r >= self.total_rows => {
                if self.total_rows > 0 {
                    Some(self.total_rows - 1)
                } else {
                    None
                }
            }
            other => other,
        };

        // Ensure selected row is visible
        if let Some(sel) = self.selected {
            self.ensure_visible(sel);
        }
    }

    /// Select the next row
    pub fn select_next(&mut self) {
        let new_sel = match self.selected {
            Some(sel) => {
                if sel + 1 < self.total_rows {
                    Some(sel + 1)
                } else {
                    Some(sel)
                }
            }
            None if self.total_rows > 0 => Some(0),
            None => None,
        };
        self.set_selected(new_sel);
    }

    /// Select the previous row
    pub fn select_prev(&mut self) {
        let new_sel = match self.selected {
            Some(sel) if sel > 0 => Some(sel - 1),
            Some(sel) => Some(sel),
            None if self.total_rows > 0 => Some(0),
            None => None,
        };
        self.set_selected(new_sel);
    }

    /// Scroll down by one row
    pub fn scroll_down(&mut self) {
        let max_offset = self.max_offset();
        if self.offset < max_offset {
            self.offset += 1;
        }
    }

    /// Scroll up by one row
    pub fn scroll_up(&mut self) {
        self.offset = self.offset.saturating_sub(1);
    }

    /// Scroll down by one page
    pub fn page_down(&mut self) {
        let page_size = self.visible_rows.max(1);
        let new_offset = self.offset.saturating_add(page_size);
        self.offset = self.clamp_offset(new_offset);
    }

    /// Scroll up by one page
    pub fn page_up(&mut self) {
        let page_size = self.visible_rows.max(1);
        self.offset = self.offset.saturating_sub(page_size);
    }

    /// Jump to the first row
    pub fn home(&mut self) {
        self.offset = 0;
    }

    /// Jump to the last page
    pub fn end(&mut self) {
        self.offset = self.max_offset();
    }

    /// Ensure a specific row is visible
    pub fn ensure_visible(&mut self, row: usize) {
        if row < self.offset {
            // Row is above viewport
            self.offset = row;
        } else if row >= self.offset + self.visible_rows {
            // Row is below viewport
            self.offset = row.saturating_sub(self.visible_rows.saturating_sub(1));
        }
        // Re-clamp to ensure valid
        self.offset = self.clamp_offset(self.offset);
    }

    /// Check if scrolling is needed (content exceeds viewport)
    pub fn needs_scrollbar(&self) -> bool {
        self.total_rows > self.visible_rows
    }

    /// Calculate scrollbar position (0.0 to 1.0)
    #[allow(clippy::cast_precision_loss)]
    pub fn scrollbar_position(&self) -> f32 {
        if self.total_rows <= self.visible_rows {
            return 0.0;
        }
        let max = self.max_offset();
        if max == 0 {
            return 0.0;
        }
        self.offset as f32 / max as f32
    }

    /// Calculate scrollbar size (0.0 to 1.0)
    #[allow(clippy::cast_precision_loss)]
    pub fn scrollbar_size(&self) -> f32 {
        if self.total_rows == 0 {
            return 1.0;
        }
        (self.visible_rows as f32 / self.total_rows as f32).min(1.0)
    }

    /// Get the maximum valid offset
    fn max_offset(&self) -> usize {
        self.total_rows.saturating_sub(self.visible_rows)
    }

    /// Clamp an offset to valid bounds
    fn clamp_offset(&self, offset: usize) -> usize {
        offset.min(self.max_offset())
    }

    /// Check if a row index is currently visible
    pub fn is_visible(&self, row: usize) -> bool {
        row >= self.offset && row < self.offset + self.visible_rows
    }

    /// Convert a global row index to viewport-relative row
    pub fn to_viewport_row(&self, global_row: usize) -> Option<usize> {
        if self.is_visible(global_row) {
            Some(global_row - self.offset)
        } else {
            None
        }
    }

    /// Convert a viewport-relative row to global row index
    pub fn to_global_row(&self, viewport_row: usize) -> usize {
        self.offset + viewport_row
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f_scroll_new() {
        let scroll = ScrollState::new(100, 20);
        assert_eq!(scroll.offset(), 0);
        assert_eq!(scroll.total_rows(), 100);
        assert_eq!(scroll.visible_rows(), 20);
    }

    #[test]
    fn f_scroll_down() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.scroll_down();
        assert_eq!(scroll.offset(), 1);
    }

    #[test]
    fn f_scroll_up() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_offset(10);
        scroll.scroll_up();
        assert_eq!(scroll.offset(), 9);
    }

    #[test]
    fn f_scroll_up_at_zero() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.scroll_up();
        assert_eq!(scroll.offset(), 0, "FALSIFIED: Should not go negative");
    }

    #[test]
    fn f_scroll_down_at_max() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_offset(80); // Max offset for 100 rows, 20 visible
        scroll.scroll_down();
        assert_eq!(scroll.offset(), 80, "FALSIFIED: Should not exceed max");
    }

    #[test]
    fn f_scroll_page_down() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.page_down();
        assert_eq!(scroll.offset(), 20);
    }

    #[test]
    fn f_scroll_page_up() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_offset(50);
        scroll.page_up();
        assert_eq!(scroll.offset(), 30);
    }

    #[test]
    fn f_scroll_home() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_offset(50);
        scroll.home();
        assert_eq!(scroll.offset(), 0);
    }

    #[test]
    fn f_scroll_end() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.end();
        assert_eq!(scroll.offset(), 80);
    }

    #[test]
    fn f_scroll_ensure_visible_above() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_offset(50);
        scroll.ensure_visible(30);
        assert_eq!(scroll.offset(), 30);
    }

    #[test]
    fn f_scroll_ensure_visible_below() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_offset(0);
        scroll.ensure_visible(30);
        assert!(scroll.offset() + scroll.visible_rows() > 30);
    }

    #[test]
    fn f_scroll_ensure_visible_already() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_offset(10);
        scroll.ensure_visible(15);
        assert_eq!(
            scroll.offset(),
            10,
            "FALSIFIED: Should not change if visible"
        );
    }

    #[test]
    fn f_scroll_needs_scrollbar_yes() {
        let scroll = ScrollState::new(100, 20);
        assert!(scroll.needs_scrollbar());
    }

    #[test]
    fn f_scroll_needs_scrollbar_no() {
        let scroll = ScrollState::new(10, 20);
        assert!(!scroll.needs_scrollbar());
    }

    #[test]
    fn f_scroll_scrollbar_position() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_offset(40);
        let pos = scroll.scrollbar_position();
        assert!(pos > 0.4 && pos < 0.6);
    }

    #[test]
    fn f_scroll_scrollbar_size() {
        let scroll = ScrollState::new(100, 20);
        let size = scroll.scrollbar_size();
        assert!((size - 0.2).abs() < 0.01);
    }

    #[test]
    fn f_scroll_is_visible() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_offset(10);
        assert!(scroll.is_visible(15));
        assert!(!scroll.is_visible(5));
        assert!(!scroll.is_visible(35));
    }

    #[test]
    fn f_scroll_to_viewport_row() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_offset(10);
        assert_eq!(scroll.to_viewport_row(15), Some(5));
        assert_eq!(scroll.to_viewport_row(5), None);
    }

    #[test]
    fn f_scroll_to_global_row() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_offset(10);
        assert_eq!(scroll.to_global_row(5), 15);
    }

    #[test]
    fn f_scroll_select_next() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_selected(Some(0));
        scroll.select_next();
        assert_eq!(scroll.selected(), Some(1));
    }

    #[test]
    fn f_scroll_select_prev() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_selected(Some(5));
        scroll.select_prev();
        assert_eq!(scroll.selected(), Some(4));
    }

    #[test]
    fn f_scroll_select_prev_at_zero() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_selected(Some(0));
        scroll.select_prev();
        assert_eq!(scroll.selected(), Some(0));
    }

    #[test]
    fn f_scroll_select_next_at_end() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_selected(Some(99));
        scroll.select_next();
        assert_eq!(scroll.selected(), Some(99));
    }

    #[test]
    fn f_scroll_select_from_none() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.select_next();
        assert_eq!(scroll.selected(), Some(0));
    }

    #[test]
    fn f_scroll_empty_dataset() {
        let scroll = ScrollState::new(0, 20);
        assert_eq!(scroll.offset(), 0);
        assert!(!scroll.needs_scrollbar());
    }

    #[test]
    fn f_scroll_set_total_rows_shrink() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_offset(80);
        scroll.set_selected(Some(90));
        scroll.set_total_rows(50);
        assert!(scroll.offset() <= 30);
        assert!(scroll.selected().unwrap_or(0) < 50);
    }

    #[test]
    fn f_scroll_default() {
        let scroll = ScrollState::default();
        assert_eq!(scroll.offset(), 0);
        assert_eq!(scroll.total_rows(), 0);
        assert_eq!(scroll.visible_rows(), 0);
    }

    #[test]
    fn f_scroll_clone() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_offset(50);
        let cloned = scroll;
        assert_eq!(scroll.offset(), cloned.offset());
    }

    #[test]
    fn f_scroll_set_total_rows_shrink_to_zero() {
        // Test shrinking to zero rows - should clear selection
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_selected(Some(50));
        scroll.set_total_rows(0);
        assert_eq!(scroll.selected(), None);
    }

    #[test]
    fn f_scroll_set_selected_out_of_bounds_empty() {
        // Test selecting row out of bounds on empty scroll
        let mut scroll = ScrollState::new(0, 20);
        scroll.set_selected(Some(100));
        // Should be None because total_rows is 0
        assert_eq!(scroll.selected(), None);
    }

    #[test]
    fn f_scroll_set_selected_out_of_bounds_clamps() {
        // Test selecting row out of bounds - should clamp to last row
        let mut scroll = ScrollState::new(50, 20);
        scroll.set_selected(Some(100));
        // Should clamp to last valid row (49)
        assert_eq!(scroll.selected(), Some(49));
    }

    #[test]
    fn f_scroll_set_total_rows_shrink_selection_out_of_bounds() {
        // Selection at row 90, then shrink to 50 rows
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_selected(Some(90));
        scroll.set_total_rows(50);
        // Selection should be clamped to 49
        assert_eq!(scroll.selected(), Some(49));
    }

    #[test]
    fn f_scroll_page_up_at_zero() {
        let mut scroll = ScrollState::new(100, 20);
        scroll.page_up();
        assert_eq!(scroll.offset(), 0);
    }

    #[test]
    fn f_scroll_select_prev_from_none() {
        let mut scroll = ScrollState::new(100, 20);
        // Select prev with no selection should select first row
        scroll.select_prev();
        assert_eq!(scroll.selected(), Some(0));
    }

    #[test]
    fn f_scroll_select_next_empty() {
        // Select next on empty dataset
        let mut scroll = ScrollState::new(0, 20);
        scroll.select_next();
        // Should still be None since total_rows is 0
        assert_eq!(scroll.selected(), None);
    }

    #[test]
    fn f_scroll_select_prev_empty() {
        // Select prev on empty dataset
        let mut scroll = ScrollState::new(0, 20);
        scroll.select_prev();
        // Should still be None since total_rows is 0
        assert_eq!(scroll.selected(), None);
    }

    #[test]
    fn f_scroll_scrollbar_position_small_content() {
        // When total_rows <= visible_rows
        let scroll = ScrollState::new(10, 20);
        let pos = scroll.scrollbar_position();
        assert!((pos - 0.0).abs() < 0.001);
    }

    #[test]
    fn f_scroll_scrollbar_position_max_zero() {
        // When max_offset is 0 (total_rows == visible_rows)
        let scroll = ScrollState::new(20, 20);
        let pos = scroll.scrollbar_position();
        assert!((pos - 0.0).abs() < 0.001);
    }

    #[test]
    fn f_scroll_scrollbar_size_empty() {
        // When total_rows is 0
        let scroll = ScrollState::new(0, 20);
        let size = scroll.scrollbar_size();
        assert!((size - 1.0).abs() < 0.001);
    }

    #[test]
    fn f_scroll_set_total_rows_with_selection_clamps() {
        // When selection is set and total_rows shrinks
        let mut scroll = ScrollState::new(100, 20);
        scroll.set_selected(Some(80));
        // Shrink to 50 rows - selection should clamp to 49
        scroll.set_total_rows(50);
        assert_eq!(scroll.selected(), Some(49));
    }
}
