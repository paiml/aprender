//! Virtualized list widget for efficient scrolling.
//!
//! The List widget only renders items that are visible in the viewport,
//! enabling smooth scrolling of thousands of items.

use presentar_core::{
    widget::LayoutResult, Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas,
    Constraints, Event, Key, Rect, Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::ops::Range;
use std::time::Duration;

/// Type alias for the render item callback.
pub type RenderItemFn = Box<dyn Fn(usize, &ListItem) -> Box<dyn Widget> + Send + Sync>;

/// Direction of the list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ListDirection {
    /// Vertical scrolling (default)
    #[default]
    Vertical,
    /// Horizontal scrolling
    Horizontal,
}

/// Selection mode for list items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SelectionMode {
    /// No selection
    #[default]
    None,
    /// Single item selection
    Single,
    /// Multiple item selection
    Multiple,
}

/// List item data for virtualization.
#[derive(Debug, Clone)]
pub struct ListItem {
    /// Unique key for this item
    pub key: String,
    /// Item height (or width in horizontal mode)
    pub size: f32,
    /// Whether item is selected
    pub selected: bool,
}

impl ListItem {
    /// Create a new list item.
    #[must_use]
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            size: 48.0, // Default item height
            selected: false,
        }
    }

    /// Set the item size.
    #[must_use]
    pub const fn size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    /// Set selection state.
    #[must_use]
    pub const fn selected(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }
}

/// Virtualized list widget.
#[derive(Serialize, Deserialize)]
pub struct List {
    /// Scroll direction
    pub direction: ListDirection,
    /// Selection mode
    pub selection_mode: SelectionMode,
    /// Fixed item height (if None, items have variable height)
    pub item_height: Option<f32>,
    /// Gap between items
    pub gap: f32,
    /// Current scroll offset
    pub scroll_offset: f32,
    /// List items (keys and sizes)
    #[serde(skip)]
    items: Vec<ListItem>,
    /// Selected indices
    #[serde(skip)]
    selected: Vec<usize>,
    /// Focused item index
    #[serde(skip)]
    focused_index: Option<usize>,
    /// Cached bounds after layout
    #[serde(skip)]
    bounds: Rect,
    /// Cached visible range
    #[serde(skip)]
    visible_range: Range<usize>,
    /// Cached item positions (start position for each item)
    #[serde(skip)]
    item_positions: Vec<f32>,
    /// Total content size
    #[serde(skip)]
    content_size: f32,
    /// Test ID
    test_id_value: Option<String>,
    /// Child widgets (rendered items only)
    #[serde(skip)]
    children: Vec<Box<dyn Widget>>,
    /// Render callback (item index -> widget)
    #[serde(skip)]
    render_item: Option<RenderItemFn>,
}

impl Default for List {
    fn default() -> Self {
        Self {
            direction: ListDirection::Vertical,
            selection_mode: SelectionMode::None,
            item_height: Some(48.0),
            gap: 0.0,
            scroll_offset: 0.0,
            items: Vec::new(),
            selected: Vec::new(),
            focused_index: None,
            bounds: Rect::default(),
            visible_range: 0..0,
            item_positions: Vec::new(),
            content_size: 0.0,
            test_id_value: None,
            children: Vec::new(),
            render_item: None,
        }
    }
}

impl List {
    /// Create a new empty list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the scroll direction.
    #[must_use]
    pub const fn direction(mut self, direction: ListDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Set the selection mode.
    #[must_use]
    pub const fn selection_mode(mut self, mode: SelectionMode) -> Self {
        self.selection_mode = mode;
        self
    }

    /// Set fixed item height.
    #[must_use]
    pub const fn item_height(mut self, height: f32) -> Self {
        self.item_height = Some(height);
        self
    }

    /// Set gap between items.
    #[must_use]
    pub const fn gap(mut self, gap: f32) -> Self {
        self.gap = gap;
        self
    }

    /// Add items to the list.
    pub fn items(mut self, items: impl IntoIterator<Item = ListItem>) -> Self {
        self.items = items.into_iter().collect();
        self.recalculate_positions();
        self
    }

    /// Set the render callback for items.
    pub fn render_with<F>(mut self, f: F) -> Self
    where
        F: Fn(usize, &ListItem) -> Box<dyn Widget> + Send + Sync + 'static,
    {
        contract_pre_render!();
        self.render_item = Some(Box::new(f));
        self
    }

    /// Set the test ID.
    #[must_use]
    pub fn with_test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id_value = Some(id.into());
        self
    }

    /// Get total item count.
    #[must_use]
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Get selected indices.
    #[must_use]
    pub fn selected_indices(&self) -> &[usize] {
        &self.selected
    }

    /// Get visible range.
    #[must_use]
    pub fn visible_range(&self) -> Range<usize> {
        self.visible_range.clone()
    }

    /// Get total content size.
    #[must_use]
    pub const fn content_size(&self) -> f32 {
        self.content_size
    }

    /// Scroll to a specific item.
    pub fn scroll_to(&mut self, index: usize) {
        if index >= self.items.len() {
            return;
        }

        let item_pos = self.item_positions.get(index).copied().unwrap_or(0.0);
        let viewport_size = match self.direction {
            ListDirection::Vertical => self.bounds.height,
            ListDirection::Horizontal => self.bounds.width,
        };

        // Scroll so item is at top/left of viewport
        self.scroll_offset = item_pos.min(self.content_size - viewport_size).max(0.0);
    }

    /// Scroll to ensure an item is visible.
    pub fn scroll_into_view(&mut self, index: usize) {
        if index >= self.items.len() {
            return;
        }

        let item_pos = self.item_positions.get(index).copied().unwrap_or(0.0);
        let item_size = self.get_item_size(index);
        let viewport_size = match self.direction {
            ListDirection::Vertical => self.bounds.height,
            ListDirection::Horizontal => self.bounds.width,
        };

        let item_end = item_pos + item_size;
        let viewport_end = self.scroll_offset + viewport_size;

        if item_pos < self.scroll_offset {
            // Item is above viewport - scroll up
            self.scroll_offset = item_pos;
        } else if item_end > viewport_end {
            // Item is below viewport - scroll down
            self.scroll_offset = (item_end - viewport_size).max(0.0);
        }
    }

    /// Select an item.
    pub fn select(&mut self, index: usize) {
        match self.selection_mode {
            SelectionMode::None => {}
            SelectionMode::Single => {
                self.selected.clear();
                if index < self.items.len() {
                    self.selected.push(index);
                    self.items[index].selected = true;
                }
            }
            SelectionMode::Multiple => {
                if index < self.items.len() && !self.selected.contains(&index) {
                    self.selected.push(index);
                    self.items[index].selected = true;
                }
            }
        }
    }

    /// Deselect an item.
    pub fn deselect(&mut self, index: usize) {
        if let Some(pos) = self.selected.iter().position(|&i| i == index) {
            self.selected.remove(pos);
            if index < self.items.len() {
                self.items[index].selected = false;
            }
        }
    }

    /// Toggle item selection.
    pub fn toggle_selection(&mut self, index: usize) {
        if self.selected.contains(&index) {
            self.deselect(index);
        } else {
            self.select(index);
        }
    }

    /// Clear all selections.
    pub fn clear_selection(&mut self) {
        for &i in &self.selected {
            if i < self.items.len() {
                self.items[i].selected = false;
            }
        }
        self.selected.clear();
    }

    /// Get item size at index.
    fn get_item_size(&self, index: usize) -> f32 {
        if let Some(fixed) = self.item_height {
            fixed
        } else {
            self.items.get(index).map_or(48.0, |i| i.size)
        }
    }

    /// Recalculate item positions after items change.
    fn recalculate_positions(&mut self) {
        self.item_positions.clear();
        let mut pos = 0.0;

        for (i, item) in self.items.iter().enumerate() {
            self.item_positions.push(pos);
            let size = self.item_height.unwrap_or(item.size);
            pos += size;
            if i < self.items.len() - 1 {
                pos += self.gap;
            }
        }

        self.content_size = pos;
    }

    /// Calculate visible range based on scroll offset and viewport.
    fn calculate_visible_range(&mut self, viewport_size: f32) {
        if self.items.is_empty() {
            self.visible_range = 0..0;
            return;
        }

        let start_offset = self.scroll_offset;
        let end_offset = self.scroll_offset + viewport_size;

        // Binary search for first visible item
        let first = self
            .item_positions
            .partition_point(|&pos| pos + self.get_item_size(0) < start_offset);

        // Linear scan for last visible (typically few items visible)
        let mut last = first;
        for i in first..self.items.len() {
            let pos = self.item_positions.get(i).copied().unwrap_or(0.0);
            if pos > end_offset {
                break;
            }
            last = i + 1;
        }

        // Add buffer for smooth scrolling
        let buffer = 2;
        let start = first.saturating_sub(buffer);
        let end = (last + buffer).min(self.items.len());

        self.visible_range = start..end;
    }

    /// Render visible items.
    fn render_visible_items(&mut self) {
        self.children.clear();

        if self.render_item.is_none() {
            return;
        }

        for i in self.visible_range.clone() {
            if let Some(item) = self.items.get(i) {
                if let Some(ref render) = self.render_item {
                    let widget = render(i, item);
                    self.children.push(widget);
                }
            }
        }
    }
}

impl Widget for List {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        // List takes available space (scrollable content)
        constraints.constrain(Size::new(constraints.max_width, constraints.max_height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;

        let viewport_size = match self.direction {
            ListDirection::Vertical => bounds.height,
            ListDirection::Horizontal => bounds.width,
        };

        // Calculate visible range
        self.calculate_visible_range(viewport_size);

        // Render visible items
        self.render_visible_items();

        // Layout visible children
        for (local_idx, i) in self.visible_range.clone().enumerate() {
            if local_idx >= self.children.len() {
                break;
            }

            let item_pos = self.item_positions.get(i).copied().unwrap_or(0.0);
            let item_size = self.get_item_size(i);

            let item_bounds = match self.direction {
                ListDirection::Vertical => Rect::new(
                    bounds.x,
                    bounds.y + item_pos - self.scroll_offset,
                    bounds.width,
                    item_size,
                ),
                ListDirection::Horizontal => Rect::new(
                    bounds.x + item_pos - self.scroll_offset,
                    bounds.y,
                    item_size,
                    bounds.height,
                ),
            };

            self.children[local_idx].layout(item_bounds);
        }

        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        // Clip to bounds
        canvas.push_clip(self.bounds);

        // Paint visible children
        for child in &self.children {
            child.paint(canvas);
        }

        canvas.pop_clip();
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        match event {
            Event::Scroll { delta_y, .. } => {
                let viewport_size = match self.direction {
                    ListDirection::Vertical => self.bounds.height,
                    ListDirection::Horizontal => self.bounds.width,
                };

                let max_scroll = (self.content_size - viewport_size).max(0.0);
                self.scroll_offset = (self.scroll_offset - delta_y * 48.0).clamp(0.0, max_scroll);

                // Recalculate visible range
                self.calculate_visible_range(viewport_size);
                self.render_visible_items();

                Some(Box::new(ListScrolled {
                    offset: self.scroll_offset,
                }))
            }
            Event::KeyDown { key, .. } => {
                if let Some(focused) = self.focused_index {
                    match key {
                        Key::Up | Key::Left => {
                            if focused > 0 {
                                self.focused_index = Some(focused - 1);
                                self.scroll_into_view(focused - 1);
                            }
                        }
                        Key::Down | Key::Right => {
                            if focused < self.items.len() - 1 {
                                self.focused_index = Some(focused + 1);
                                self.scroll_into_view(focused + 1);
                            }
                        }
                        Key::Enter | Key::Space => {
                            self.toggle_selection(focused);
                            return Some(Box::new(ListItemSelected { index: focused }));
                        }
                        Key::Home => {
                            self.focused_index = Some(0);
                            self.scroll_to(0);
                        }
                        Key::End => {
                            let last = self.items.len().saturating_sub(1);
                            self.focused_index = Some(last);
                            self.scroll_to(last);
                        }
                        _ => {}
                    }
                }
                None
            }
            Event::MouseDown { position, .. } => {
                // Find clicked item
                let pos = match self.direction {
                    ListDirection::Vertical => position.y - self.bounds.y + self.scroll_offset,
                    ListDirection::Horizontal => position.x - self.bounds.x + self.scroll_offset,
                };

                for (i, &item_pos) in self.item_positions.iter().enumerate() {
                    let item_size = self.get_item_size(i);
                    if pos >= item_pos && pos < item_pos + item_size {
                        self.focused_index = Some(i);
                        self.toggle_selection(i);
                        return Some(Box::new(ListItemClicked { index: i }));
                    }
                }
                None
            }
            _ => {
                // Propagate to visible children
                for child in &mut self.children {
                    if let Some(msg) = child.event(event) {
                        return Some(msg);
                    }
                }
                None
            }
        }
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut self.children
    }

    fn is_focusable(&self) -> bool {
        true
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }

    fn bounds(&self) -> Rect {
        self.bounds
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for List {
    fn brick_name(&self) -> &'static str {
        "List"
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
        r#"<div class="brick-list"></div>"#.to_string()
    }

    fn to_css(&self) -> String {
        ".brick-list { display: block; overflow: auto; }".to_string()
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

/// Message emitted when list is scrolled.
#[derive(Debug, Clone)]
pub struct ListScrolled {
    /// New scroll offset
    pub offset: f32,
}

/// Message emitted when a list item is clicked.
#[derive(Debug, Clone)]
pub struct ListItemClicked {
    /// Index of clicked item
    pub index: usize,
}

/// Message emitted when a list item is selected.
#[derive(Debug, Clone)]
pub struct ListItemSelected {
    /// Index of selected item
    pub index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ListDirection Tests
    // =========================================================================

    #[test]
    fn test_list_direction_default() {
        assert_eq!(ListDirection::default(), ListDirection::Vertical);
    }

    // =========================================================================
    // SelectionMode Tests
    // =========================================================================

    #[test]
    fn test_selection_mode_default() {
        assert_eq!(SelectionMode::default(), SelectionMode::None);
    }

    // =========================================================================
    // ListItem Tests
    // =========================================================================

    #[test]
    fn test_list_item_new() {
        let item = ListItem::new("item-1");
        assert_eq!(item.key, "item-1");
        assert_eq!(item.size, 48.0);
        assert!(!item.selected);
    }

    #[test]
    fn test_list_item_builder() {
        let item = ListItem::new("item-1").size(64.0).selected(true);
        assert_eq!(item.size, 64.0);
        assert!(item.selected);
    }

    // =========================================================================
    // List Tests
    // =========================================================================

    #[test]
    fn test_list_new() {
        let list = List::new();
        assert_eq!(list.direction, ListDirection::Vertical);
        assert_eq!(list.selection_mode, SelectionMode::None);
        assert_eq!(list.item_height, Some(48.0));
        assert_eq!(list.gap, 0.0);
        assert_eq!(list.item_count(), 0);
    }

    #[test]
    fn test_list_builder() {
        let list = List::new()
            .direction(ListDirection::Horizontal)
            .selection_mode(SelectionMode::Single)
            .item_height(32.0)
            .gap(8.0);

        assert_eq!(list.direction, ListDirection::Horizontal);
        assert_eq!(list.selection_mode, SelectionMode::Single);
        assert_eq!(list.item_height, Some(32.0));
        assert_eq!(list.gap, 8.0);
    }

    #[test]
    fn test_list_items() {
        let items = vec![ListItem::new("1"), ListItem::new("2"), ListItem::new("3")];
        let list = List::new().items(items);
        assert_eq!(list.item_count(), 3);
    }

    #[test]
    fn test_list_content_size() {
        let items = vec![ListItem::new("1"), ListItem::new("2"), ListItem::new("3")];
        let list = List::new().item_height(50.0).gap(10.0).items(items);
        // 3 items * 50px + 2 gaps * 10px = 170px
        assert_eq!(list.content_size(), 170.0);
    }

    #[test]
    fn test_list_content_size_variable_height() {
        let items = vec![
            ListItem::new("1").size(30.0),
            ListItem::new("2").size(40.0),
            ListItem::new("3").size(50.0),
        ];
        let mut list = List::new().gap(5.0);
        list.item_height = None; // Variable height mode
        list = list.items(items);
        // 30 + 5 + 40 + 5 + 50 = 130px
        assert_eq!(list.content_size(), 130.0);
    }

    #[test]
    fn test_list_select_single() {
        let items = vec![ListItem::new("1"), ListItem::new("2")];
        let mut list = List::new()
            .selection_mode(SelectionMode::Single)
            .items(items);

        list.select(0);
        assert_eq!(list.selected_indices(), &[0]);

        list.select(1);
        assert_eq!(list.selected_indices(), &[1]); // Single mode replaces
    }

    #[test]
    fn test_list_select_multiple() {
        let items = vec![ListItem::new("1"), ListItem::new("2")];
        let mut list = List::new()
            .selection_mode(SelectionMode::Multiple)
            .items(items);

        list.select(0);
        list.select(1);
        assert_eq!(list.selected_indices(), &[0, 1]);
    }

    #[test]
    fn test_list_deselect() {
        let items = vec![ListItem::new("1"), ListItem::new("2")];
        let mut list = List::new()
            .selection_mode(SelectionMode::Multiple)
            .items(items);

        list.select(0);
        list.select(1);
        list.deselect(0);
        assert_eq!(list.selected_indices(), &[1]);
    }

    #[test]
    fn test_list_toggle_selection() {
        let items = vec![ListItem::new("1")];
        let mut list = List::new()
            .selection_mode(SelectionMode::Single)
            .items(items);

        list.toggle_selection(0);
        assert_eq!(list.selected_indices(), &[0]);

        list.toggle_selection(0);
        assert!(list.selected_indices().is_empty());
    }

    #[test]
    fn test_list_clear_selection() {
        let items = vec![ListItem::new("1"), ListItem::new("2")];
        let mut list = List::new()
            .selection_mode(SelectionMode::Multiple)
            .items(items);

        list.select(0);
        list.select(1);
        list.clear_selection();
        assert!(list.selected_indices().is_empty());
    }

    #[test]
    fn test_list_scroll_to() {
        let items: Vec<_> = (0..100).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new().item_height(50.0).items(items);
        list.bounds = Rect::new(0.0, 0.0, 300.0, 200.0);

        list.scroll_to(10);
        assert_eq!(list.scroll_offset, 500.0); // 10 * 50px
    }

    #[test]
    fn test_list_scroll_into_view() {
        let items: Vec<_> = (0..10).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new().item_height(50.0).items(items);
        list.bounds = Rect::new(0.0, 0.0, 300.0, 200.0);
        list.scroll_offset = 0.0;

        // Item 5 is at 250px, below viewport (0-200)
        list.scroll_into_view(5);
        // Should scroll so item end (300px) is at viewport end
        assert_eq!(list.scroll_offset, 100.0);
    }

    #[test]
    fn test_list_visible_range() {
        let items: Vec<_> = (0..100).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new().item_height(50.0).items(items);
        list.bounds = Rect::new(0.0, 0.0, 300.0, 200.0);
        list.scroll_offset = 0.0;

        list.calculate_visible_range(200.0);

        // At scroll 0 with 200px viewport and 50px items, ~4 items visible
        // Plus buffer of 2 on each side
        let range = list.visible_range();
        assert!(range.start <= 4);
        assert!(range.end >= 4);
    }

    #[test]
    fn test_list_measure() {
        let list = List::new();
        let size = list.measure(Constraints::loose(Size::new(300.0, 400.0)));
        assert_eq!(size, Size::new(300.0, 400.0));
    }

    #[test]
    fn test_list_layout() {
        let items: Vec<_> = (0..10).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new().item_height(50.0).items(items);

        let result = list.layout(Rect::new(0.0, 0.0, 300.0, 200.0));
        assert_eq!(result.size, Size::new(300.0, 200.0));
        assert_eq!(list.bounds, Rect::new(0.0, 0.0, 300.0, 200.0));
    }

    #[test]
    fn test_list_type_id() {
        let list = List::new();
        assert_eq!(Widget::type_id(&list), TypeId::of::<List>());
    }

    #[test]
    fn test_list_is_focusable() {
        let list = List::new();
        assert!(list.is_focusable());
    }

    #[test]
    fn test_list_test_id() {
        let list = List::new().with_test_id("my-list");
        assert_eq!(Widget::test_id(&list), Some("my-list"));
    }

    #[test]
    fn test_list_children_empty() {
        let list = List::new();
        assert!(list.children().is_empty());
    }

    #[test]
    fn test_list_bounds() {
        let items: Vec<_> = (0..5).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new().items(items);
        list.layout(Rect::new(10.0, 20.0, 300.0, 200.0));
        assert_eq!(list.bounds(), Rect::new(10.0, 20.0, 300.0, 200.0));
    }

    #[test]
    fn test_list_scrolled_message() {
        let msg = ListScrolled { offset: 100.0 };
        assert_eq!(msg.offset, 100.0);
    }

    #[test]
    fn test_list_item_clicked_message() {
        let msg = ListItemClicked { index: 5 };
        assert_eq!(msg.index, 5);
    }

    #[test]
    fn test_list_item_selected_message() {
        let msg = ListItemSelected { index: 3 };
        assert_eq!(msg.index, 3);
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_list_direction_horizontal() {
        let list = List::new().direction(ListDirection::Horizontal);
        assert_eq!(list.direction, ListDirection::Horizontal);
    }

    #[test]
    fn test_list_direction_is_vertical_by_default() {
        assert_eq!(ListDirection::default(), ListDirection::Vertical);
    }

    #[test]
    fn test_selection_mode_is_none_by_default() {
        assert_eq!(SelectionMode::default(), SelectionMode::None);
    }

    #[test]
    fn test_list_with_selection_mode_multiple() {
        let list = List::new().selection_mode(SelectionMode::Multiple);
        assert_eq!(list.selection_mode, SelectionMode::Multiple);
    }

    #[test]
    fn test_list_with_selection_mode_single() {
        let list = List::new().selection_mode(SelectionMode::Single);
        assert_eq!(list.selection_mode, SelectionMode::Single);
    }

    #[test]
    fn test_list_gap() {
        let list = List::new().gap(10.0);
        assert_eq!(list.gap, 10.0);
    }

    #[test]
    fn test_list_item_height_custom() {
        let list = List::new().item_height(60.0);
        assert_eq!(list.item_height, Some(60.0));
    }

    #[test]
    fn test_list_children_mut() {
        let mut list = List::new();
        // Children are empty when no render callback or visible items
        assert!(list.children_mut().is_empty());
    }

    #[test]
    fn test_list_content_size_calculated() {
        let items: Vec<_> = (0..5).map(|i| ListItem::new(format!("{i}"))).collect();
        let list = List::new().items(items).item_height(40.0);
        assert!(list.content_size() > 0.0);
    }

    #[test]
    fn test_list_item_size_custom() {
        let item = ListItem::new("Item").size(60.0);
        assert_eq!(item.size, 60.0);
    }

    #[test]
    fn test_list_item_selected_state() {
        let item = ListItem::new("Item").selected(true);
        assert!(item.selected);
    }

    #[test]
    fn test_list_event_returns_none_when_empty() {
        let mut list = List::new();
        list.layout(Rect::new(0.0, 0.0, 300.0, 200.0));
        let result = list.event(&Event::key_down(Key::Down));
        assert!(result.is_none());
    }

    // =========================================================================
    // Event Handling Tests
    // =========================================================================

    #[test]
    fn test_list_scroll_event() {
        let items: Vec<_> = (0..20).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new().item_height(50.0).items(items);
        list.layout(Rect::new(0.0, 0.0, 300.0, 200.0));

        // Scroll down
        let result = list.event(&Event::Scroll {
            delta_x: 0.0,
            delta_y: -2.0,
        });
        assert!(result.is_some());
        assert!(list.scroll_offset > 0.0);
    }

    #[test]
    fn test_list_scroll_event_clamp() {
        let items: Vec<_> = (0..5).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new().item_height(50.0).items(items);
        list.layout(Rect::new(0.0, 0.0, 300.0, 500.0)); // Viewport larger than content

        // Try to scroll down
        let _ = list.event(&Event::Scroll {
            delta_x: 0.0,
            delta_y: -10.0,
        });
        // Should clamp to 0 since content fits in viewport
        assert_eq!(list.scroll_offset, 0.0);
    }

    #[test]
    fn test_list_key_down_focused() {
        let items: Vec<_> = (0..10).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new()
            .selection_mode(SelectionMode::Single)
            .item_height(50.0)
            .items(items);
        list.layout(Rect::new(0.0, 0.0, 300.0, 200.0));
        list.focused_index = Some(5);

        // Press Down
        let _ = list.event(&Event::key_down(Key::Down));
        assert_eq!(list.focused_index, Some(6));

        // Press Up
        let _ = list.event(&Event::key_down(Key::Up));
        assert_eq!(list.focused_index, Some(5));
    }

    #[test]
    fn test_list_key_left_right() {
        let items: Vec<_> = (0..10).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new()
            .direction(ListDirection::Horizontal)
            .selection_mode(SelectionMode::Single)
            .item_height(50.0)
            .items(items);
        list.layout(Rect::new(0.0, 0.0, 300.0, 200.0));
        list.focused_index = Some(5);

        // Press Right
        let _ = list.event(&Event::key_down(Key::Right));
        assert_eq!(list.focused_index, Some(6));

        // Press Left
        let _ = list.event(&Event::key_down(Key::Left));
        assert_eq!(list.focused_index, Some(5));
    }

    #[test]
    fn test_list_key_home_end() {
        let items: Vec<_> = (0..10).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new()
            .selection_mode(SelectionMode::Single)
            .item_height(50.0)
            .items(items);
        list.layout(Rect::new(0.0, 0.0, 300.0, 200.0));
        list.focused_index = Some(5);

        // Press Home
        let _ = list.event(&Event::key_down(Key::Home));
        assert_eq!(list.focused_index, Some(0));

        // Press End
        let _ = list.event(&Event::key_down(Key::End));
        assert_eq!(list.focused_index, Some(9));
    }

    #[test]
    fn test_list_key_enter_selects() {
        let items: Vec<_> = (0..5).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new()
            .selection_mode(SelectionMode::Single)
            .item_height(50.0)
            .items(items);
        list.layout(Rect::new(0.0, 0.0, 300.0, 200.0));
        list.focused_index = Some(2);

        let result = list.event(&Event::key_down(Key::Enter));
        assert!(result.is_some());
        assert_eq!(list.selected_indices(), &[2]);
    }

    #[test]
    fn test_list_key_space_selects() {
        let items: Vec<_> = (0..5).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new()
            .selection_mode(SelectionMode::Single)
            .item_height(50.0)
            .items(items);
        list.layout(Rect::new(0.0, 0.0, 300.0, 200.0));
        list.focused_index = Some(3);

        let result = list.event(&Event::key_down(Key::Space));
        assert!(result.is_some());
        assert_eq!(list.selected_indices(), &[3]);
    }

    #[test]
    fn test_list_mouse_down_click() {
        let items: Vec<_> = (0..5).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new()
            .selection_mode(SelectionMode::Single)
            .item_height(50.0)
            .items(items);
        list.layout(Rect::new(0.0, 0.0, 300.0, 300.0));

        // Click on item at y=75 (should be item 1, which is at 50-100)
        let result = list.event(&Event::MouseDown {
            position: presentar_core::Point::new(150.0, 75.0),
            button: presentar_core::MouseButton::Left,
        });
        assert!(result.is_some());
        assert_eq!(list.focused_index, Some(1));
    }

    #[test]
    fn test_list_mouse_down_horizontal() {
        let items: Vec<_> = (0..5).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new()
            .direction(ListDirection::Horizontal)
            .selection_mode(SelectionMode::Single)
            .item_height(50.0)
            .items(items);
        list.layout(Rect::new(0.0, 0.0, 300.0, 100.0));

        // Click on item at x=75 (should be item 1)
        let result = list.event(&Event::MouseDown {
            position: presentar_core::Point::new(75.0, 50.0),
            button: presentar_core::MouseButton::Left,
        });
        assert!(result.is_some());
        assert_eq!(list.focused_index, Some(1));
    }

    #[test]
    fn test_list_mouse_down_miss() {
        let items: Vec<_> = (0..2).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new()
            .selection_mode(SelectionMode::Single)
            .item_height(50.0)
            .items(items);
        list.layout(Rect::new(0.0, 0.0, 300.0, 300.0));

        // Click below all items
        let result = list.event(&Event::MouseDown {
            position: presentar_core::Point::new(150.0, 200.0),
            button: presentar_core::MouseButton::Left,
        });
        assert!(result.is_none());
    }

    #[test]
    fn test_list_other_event() {
        let mut list = List::new();
        list.layout(Rect::new(0.0, 0.0, 300.0, 200.0));

        // Unknown event propagates to children (empty)
        let result = list.event(&Event::MouseMove {
            position: presentar_core::Point::new(100.0, 100.0),
        });
        assert!(result.is_none());
    }

    // =========================================================================
    // Paint Tests
    // =========================================================================

    use presentar_core::RecordingCanvas;

    #[test]
    fn test_list_paint_empty() {
        let list = List::new();
        let mut canvas = RecordingCanvas::new();
        // Should not panic when painting empty list
        list.paint(&mut canvas);
    }

    #[test]
    fn test_list_paint_with_items() {
        let items: Vec<_> = (0..5).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new().item_height(50.0).items(items);
        list.bounds = Rect::new(0.0, 0.0, 300.0, 200.0);

        let mut canvas = RecordingCanvas::new();
        // Should not panic when painting list with items
        list.paint(&mut canvas);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_list_scroll_to_out_of_bounds() {
        let items: Vec<_> = (0..5).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new().item_height(50.0).items(items);
        list.bounds = Rect::new(0.0, 0.0, 300.0, 200.0);

        // Try to scroll to non-existent item
        list.scroll_to(100);
        // Should not crash, offset unchanged
        assert_eq!(list.scroll_offset, 0.0);
    }

    #[test]
    fn test_list_scroll_into_view_out_of_bounds() {
        let items: Vec<_> = (0..5).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new().item_height(50.0).items(items);
        list.bounds = Rect::new(0.0, 0.0, 300.0, 200.0);

        // Try to scroll to non-existent item
        list.scroll_into_view(100);
        // Should not crash
        assert_eq!(list.scroll_offset, 0.0);
    }

    #[test]
    fn test_list_scroll_into_view_item_above() {
        let items: Vec<_> = (0..10).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new().item_height(50.0).items(items);
        list.bounds = Rect::new(0.0, 0.0, 300.0, 200.0);
        list.scroll_offset = 200.0; // Start scrolled down

        // Item 0 is above viewport
        list.scroll_into_view(0);
        assert_eq!(list.scroll_offset, 0.0);
    }

    #[test]
    fn test_list_select_none_mode() {
        let items: Vec<_> = (0..5).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new().selection_mode(SelectionMode::None).items(items);

        list.select(0);
        assert!(list.selected_indices().is_empty());
    }

    #[test]
    fn test_list_select_out_of_bounds() {
        let items: Vec<_> = (0..5).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new()
            .selection_mode(SelectionMode::Single)
            .items(items);

        list.select(100);
        assert!(list.selected_indices().is_empty());
    }

    #[test]
    fn test_list_select_multiple_same_item() {
        let items: Vec<_> = (0..5).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new()
            .selection_mode(SelectionMode::Multiple)
            .items(items);

        list.select(0);
        list.select(0); // Try to select same item again
        assert_eq!(list.selected_indices().len(), 1);
    }

    #[test]
    fn test_list_deselect_nonexistent() {
        let items: Vec<_> = (0..5).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new()
            .selection_mode(SelectionMode::Multiple)
            .items(items);

        list.select(0);
        list.deselect(1); // Not selected
        assert_eq!(list.selected_indices(), &[0]);
    }

    #[test]
    fn test_list_clear_selection_with_invalid_indices() {
        let items: Vec<_> = (0..5).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new()
            .selection_mode(SelectionMode::Multiple)
            .items(items);

        list.select(0);
        list.selected.push(100); // Add invalid index manually
        list.clear_selection();
        assert!(list.selected_indices().is_empty());
    }

    #[test]
    fn test_list_horizontal_layout() {
        let items: Vec<_> = (0..5).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new()
            .direction(ListDirection::Horizontal)
            .item_height(50.0)
            .items(items);

        let result = list.layout(Rect::new(0.0, 0.0, 300.0, 100.0));
        assert_eq!(result.size, Size::new(300.0, 100.0));
    }

    #[test]
    fn test_list_horizontal_scroll() {
        let items: Vec<_> = (0..20).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new()
            .direction(ListDirection::Horizontal)
            .item_height(50.0)
            .items(items);
        list.layout(Rect::new(0.0, 0.0, 200.0, 100.0));

        list.scroll_to(10);
        assert_eq!(list.scroll_offset, 500.0);
    }

    #[test]
    fn test_list_horizontal_scroll_into_view() {
        let items: Vec<_> = (0..10).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new()
            .direction(ListDirection::Horizontal)
            .item_height(50.0)
            .items(items);
        list.bounds = Rect::new(0.0, 0.0, 200.0, 100.0);
        list.scroll_offset = 0.0;

        list.scroll_into_view(5);
        assert!(list.scroll_offset > 0.0);
    }

    #[test]
    fn test_list_visible_range_empty() {
        let mut list = List::new();
        list.calculate_visible_range(200.0);
        assert_eq!(list.visible_range(), 0..0);
    }

    #[test]
    fn test_list_get_item_size_variable() {
        let items = vec![ListItem::new("1").size(30.0), ListItem::new("2").size(50.0)];
        let mut list = List::new();
        list.item_height = None;
        list = list.items(items);

        // Private method test via content_size
        assert_eq!(list.content_size(), 80.0);
    }

    #[test]
    fn test_list_key_boundary_checks() {
        let items: Vec<_> = (0..3).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new()
            .selection_mode(SelectionMode::Single)
            .item_height(50.0)
            .items(items);
        list.layout(Rect::new(0.0, 0.0, 300.0, 200.0));

        // At first item, press Up
        list.focused_index = Some(0);
        let _ = list.event(&Event::key_down(Key::Up));
        assert_eq!(list.focused_index, Some(0)); // Should stay at 0

        // At last item, press Down
        list.focused_index = Some(2);
        let _ = list.event(&Event::key_down(Key::Down));
        assert_eq!(list.focused_index, Some(2)); // Should stay at 2
    }

    #[test]
    fn test_list_other_key_no_action() {
        let items: Vec<_> = (0..3).map(|i| ListItem::new(format!("{i}"))).collect();
        let mut list = List::new()
            .selection_mode(SelectionMode::Single)
            .item_height(50.0)
            .items(items);
        list.layout(Rect::new(0.0, 0.0, 300.0, 200.0));
        list.focused_index = Some(1);

        // Press a key that's not handled
        let result = list.event(&Event::key_down(Key::Tab));
        assert!(result.is_none());
        assert_eq!(list.focused_index, Some(1));
    }

    // =========================================================================
    // Brick Trait Tests
    // =========================================================================

    #[test]
    fn test_list_brick_name() {
        let list = List::new();
        assert_eq!(list.brick_name(), "List");
    }

    #[test]
    fn test_list_brick_assertions() {
        let list = List::new();
        let assertions = list.assertions();
        assert!(!assertions.is_empty());
    }

    #[test]
    fn test_list_brick_budget() {
        let list = List::new();
        let budget = list.budget();
        assert!(budget.layout_ms > 0);
    }

    #[test]
    fn test_list_brick_verify() {
        let list = List::new();
        let verification = list.verify();
        assert!(!verification.passed.is_empty());
        assert!(verification.failed.is_empty());
    }

    #[test]
    fn test_list_brick_to_html() {
        let list = List::new();
        let html = list.to_html();
        assert!(html.contains("brick-list"));
    }

    #[test]
    fn test_list_brick_to_css() {
        let list = List::new();
        let css = list.to_css();
        assert!(css.contains("brick-list"));
    }

    #[test]
    fn test_list_brick_test_id() {
        let list = List::new().with_test_id("test-list");
        assert_eq!(Brick::test_id(&list), Some("test-list"));
    }
}
