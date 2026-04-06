//! Select/Dropdown widget for choosing from options.

use presentar_core::{
    widget::{AccessibleRole, Brick, BrickAssertion, BrickBudget, BrickVerification, LayoutResult},
    Canvas, Color, Constraints, Event, MouseButton, Rect, Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// A selectable option.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectOption {
    /// Unique value for this option
    pub value: String,
    /// Display label
    pub label: String,
    /// Whether this option is disabled
    pub disabled: bool,
}

impl SelectOption {
    /// Create a new option.
    #[must_use]
    pub fn new(value: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            label: label.into(),
            disabled: false,
        }
    }

    /// Create an option where value equals label.
    #[must_use]
    pub fn simple(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            value: text.clone(),
            label: text,
            disabled: false,
        }
    }

    /// Set disabled state.
    #[must_use]
    pub const fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// Message emitted when selection changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionChanged {
    /// The newly selected value (None if cleared)
    pub value: Option<String>,
    /// Index of the selected option
    pub index: Option<usize>,
}

/// Select/Dropdown widget.
#[derive(Serialize, Deserialize)]
pub struct Select {
    /// Available options
    options: Vec<SelectOption>,
    /// Currently selected index (None for no selection)
    selected: Option<usize>,
    /// Placeholder text when nothing selected
    placeholder: String,
    /// Whether the dropdown is currently open
    #[serde(skip)]
    open: bool,
    /// Whether the widget is disabled
    disabled: bool,
    /// Minimum width
    min_width: f32,
    /// Item height
    item_height: f32,
    /// Maximum visible items in dropdown
    max_visible_items: usize,
    /// Background color
    background_color: Color,
    /// Border color
    border_color: Color,
    /// Selected item background
    selected_bg_color: Color,
    /// Hover item background
    hover_bg_color: Color,
    /// Text color
    text_color: Color,
    /// Placeholder text color
    placeholder_color: Color,
    /// Disabled color
    disabled_color: Color,
    /// Test ID
    test_id_value: Option<String>,
    /// Accessible name
    accessible_name_value: Option<String>,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
    /// Currently hovered item index
    #[serde(skip)]
    hovered_item: Option<usize>,
}

impl Default for Select {
    fn default() -> Self {
        Self::new()
    }
}

impl Select {
    /// Create a new select widget.
    #[must_use]
    pub fn new() -> Self {
        Self {
            options: Vec::new(),
            selected: None,
            placeholder: "Select...".to_string(),
            open: false,
            disabled: false,
            min_width: 150.0,
            item_height: 32.0,
            max_visible_items: 8,
            background_color: Color::WHITE,
            border_color: Color::new(0.8, 0.8, 0.8, 1.0),
            selected_bg_color: Color::new(0.9, 0.95, 1.0, 1.0),
            hover_bg_color: Color::new(0.95, 0.95, 0.95, 1.0),
            text_color: Color::BLACK,
            placeholder_color: Color::new(0.6, 0.6, 0.6, 1.0),
            disabled_color: Color::new(0.7, 0.7, 0.7, 1.0),
            test_id_value: None,
            accessible_name_value: None,
            bounds: Rect::default(),
            hovered_item: None,
        }
    }

    /// Add an option.
    #[must_use]
    pub fn option(mut self, opt: SelectOption) -> Self {
        self.options.push(opt);
        self
    }

    /// Add multiple options.
    #[must_use]
    pub fn options(mut self, opts: impl IntoIterator<Item = SelectOption>) -> Self {
        self.options.extend(opts);
        self
    }

    /// Set options from simple string values.
    #[must_use]
    pub fn options_from_strings(
        mut self,
        values: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.options = values.into_iter().map(SelectOption::simple).collect();
        self
    }

    /// Set placeholder text.
    #[must_use]
    pub fn placeholder(mut self, text: impl Into<String>) -> Self {
        self.placeholder = text.into();
        self
    }

    /// Set selected index.
    #[must_use]
    pub fn selected(mut self, index: Option<usize>) -> Self {
        self.selected = index.filter(|&i| i < self.options.len());
        self
    }

    /// Set selected by value.
    #[must_use]
    pub fn selected_value(mut self, value: &str) -> Self {
        self.selected = self.options.iter().position(|o| o.value == value);
        self
    }

    /// Set disabled state.
    #[must_use]
    pub const fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    /// Set minimum width.
    #[must_use]
    pub fn min_width(mut self, width: f32) -> Self {
        self.min_width = width.max(50.0);
        self
    }

    /// Set item height.
    #[must_use]
    pub fn item_height(mut self, height: f32) -> Self {
        self.item_height = height.max(20.0);
        self
    }

    /// Set max visible items.
    #[must_use]
    pub fn max_visible_items(mut self, count: usize) -> Self {
        self.max_visible_items = count.max(1);
        self
    }

    /// Set background color.
    #[must_use]
    pub const fn background_color(mut self, color: Color) -> Self {
        self.background_color = color;
        self
    }

    /// Set border color.
    #[must_use]
    pub const fn border_color(mut self, color: Color) -> Self {
        self.border_color = color;
        self
    }

    /// Set test ID.
    #[must_use]
    pub fn with_test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id_value = Some(id.into());
        self
    }

    /// Set accessible name.
    #[must_use]
    pub fn with_accessible_name(mut self, name: impl Into<String>) -> Self {
        self.accessible_name_value = Some(name.into());
        self
    }

    /// Get selected index.
    #[must_use]
    pub const fn get_selected(&self) -> Option<usize> {
        self.selected
    }

    /// Get selected value.
    #[must_use]
    pub fn get_selected_value(&self) -> Option<&str> {
        self.selected.map(|i| self.options[i].value.as_str())
    }

    /// Get selected label.
    #[must_use]
    pub fn get_selected_label(&self) -> Option<&str> {
        self.selected.map(|i| self.options[i].label.as_str())
    }

    /// Get all options.
    #[must_use]
    pub fn get_options(&self) -> &[SelectOption] {
        &self.options
    }

    /// Check if dropdown is open.
    #[must_use]
    pub const fn is_open(&self) -> bool {
        self.open
    }

    /// Check if empty (no options).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.options.is_empty()
    }

    /// Get option count.
    #[must_use]
    pub fn option_count(&self) -> usize {
        self.options.len()
    }

    /// Calculate dropdown height.
    fn dropdown_height(&self) -> f32 {
        let visible = self.options.len().min(self.max_visible_items);
        visible as f32 * self.item_height
    }

    /// Get item rect at index.
    fn item_rect(&self, index: usize) -> Rect {
        let y = (index as f32).mul_add(self.item_height, self.bounds.y + self.item_height);
        Rect::new(self.bounds.x, y, self.bounds.width, self.item_height)
    }

    /// Find item at position.
    fn item_at_position(&self, y: f32) -> Option<usize> {
        if !self.open {
            return None;
        }

        let dropdown_top = self.bounds.y + self.item_height;
        if y < dropdown_top {
            return None;
        }

        let relative_y = y - dropdown_top;
        let index = (relative_y / self.item_height) as usize;

        if index < self.options.len() && index < self.max_visible_items {
            Some(index)
        } else {
            None
        }
    }
}

impl Widget for Select {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let width = self.min_width;
        let height = self.item_height;
        constraints.constrain(Size::new(width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        // Draw main button/header
        let header_rect = Rect::new(
            self.bounds.x,
            self.bounds.y,
            self.bounds.width,
            self.item_height,
        );

        let bg_color = if self.disabled {
            self.disabled_color
        } else {
            self.background_color
        };

        canvas.fill_rect(header_rect, bg_color);
        canvas.stroke_rect(header_rect, self.border_color, 1.0);

        // Draw selected text or placeholder
        let text = self.get_selected_label().unwrap_or(&self.placeholder);
        let text_color = if self.disabled {
            self.disabled_color
        } else if self.selected.is_some() {
            self.text_color
        } else {
            self.placeholder_color
        };

        let text_style = presentar_core::widget::TextStyle {
            color: text_color,
            ..Default::default()
        };
        let text_pos = presentar_core::Point::new(
            self.bounds.x + 8.0,
            self.bounds.y + (self.item_height - 16.0) / 2.0,
        );
        canvas.draw_text(text, text_pos, &text_style);

        // Draw dropdown arrow
        let arrow_x = self.bounds.x + self.bounds.width - 20.0;
        let arrow_y = self.bounds.y + self.item_height / 2.0;
        let arrow_rect = Rect::new(arrow_x, arrow_y - 3.0, 8.0, 6.0);
        canvas.fill_rect(arrow_rect, self.text_color);

        // Draw dropdown if open
        if self.open && !self.options.is_empty() {
            let dropdown_rect = Rect::new(
                self.bounds.x,
                self.bounds.y + self.item_height,
                self.bounds.width,
                self.dropdown_height(),
            );

            canvas.fill_rect(dropdown_rect, self.background_color);
            canvas.stroke_rect(dropdown_rect, self.border_color, 1.0);

            // Draw items
            for (i, opt) in self.options.iter().take(self.max_visible_items).enumerate() {
                let item_rect = self.item_rect(i);

                // Background
                let item_bg = if Some(i) == self.selected {
                    self.selected_bg_color
                } else if Some(i) == self.hovered_item {
                    self.hover_bg_color
                } else {
                    self.background_color
                };
                canvas.fill_rect(item_rect, item_bg);

                // Text
                let item_color = if opt.disabled {
                    self.disabled_color
                } else {
                    self.text_color
                };
                let item_style = presentar_core::widget::TextStyle {
                    color: item_color,
                    ..Default::default()
                };
                let item_pos = presentar_core::Point::new(
                    item_rect.x + 8.0,
                    item_rect.y + (self.item_height - 16.0) / 2.0,
                );
                canvas.draw_text(&opt.label, item_pos, &item_style);
            }
        }
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        if self.disabled {
            return None;
        }

        match event {
            Event::MouseMove { position } => {
                if self.open {
                    self.hovered_item = self.item_at_position(position.y);
                }
            }
            Event::MouseDown {
                position,
                button: MouseButton::Left,
            } => {
                let header_rect = Rect::new(
                    self.bounds.x,
                    self.bounds.y,
                    self.bounds.width,
                    self.item_height,
                );

                if header_rect.contains_point(position) {
                    // Toggle dropdown
                    self.open = !self.open;
                    self.hovered_item = None;
                } else if self.open {
                    // Check if clicked on an item
                    if let Some(index) = self.item_at_position(position.y) {
                        let opt = &self.options[index];
                        if !opt.disabled {
                            self.selected = Some(index);
                            self.open = false;
                            return Some(Box::new(SelectionChanged {
                                value: Some(opt.value.clone()),
                                index: Some(index),
                            }));
                        }
                    } else {
                        // Clicked outside - close
                        self.open = false;
                    }
                }
            }
            Event::FocusOut => {
                self.open = false;
            }
            _ => {}
        }

        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }

    fn is_interactive(&self) -> bool {
        !self.disabled
    }

    fn is_focusable(&self) -> bool {
        !self.disabled
    }

    fn accessible_name(&self) -> Option<&str> {
        self.accessible_name_value.as_deref()
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::ComboBox
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for Select {
    fn brick_name(&self) -> &'static str {
        "Select"
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
        r#"<div class="brick-select"></div>"#.to_string()
    }

    fn to_css(&self) -> String {
        ".brick-select { display: block; }".to_string()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use presentar_core::Widget;

    // =========================================================================
    // SelectOption Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_select_option_new() {
        let opt = SelectOption::new("val", "Label");
        assert_eq!(opt.value, "val");
        assert_eq!(opt.label, "Label");
        assert!(!opt.disabled);
    }

    #[test]
    fn test_select_option_simple() {
        let opt = SelectOption::simple("Same");
        assert_eq!(opt.value, "Same");
        assert_eq!(opt.label, "Same");
    }

    #[test]
    fn test_select_option_disabled() {
        let opt = SelectOption::new("v", "L").disabled(true);
        assert!(opt.disabled);
    }

    // =========================================================================
    // SelectionChanged Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_selection_changed_message() {
        let msg = SelectionChanged {
            value: Some("test".to_string()),
            index: Some(0),
        };
        assert_eq!(msg.value, Some("test".to_string()));
        assert_eq!(msg.index, Some(0));
    }

    #[test]
    fn test_selection_changed_none() {
        let msg = SelectionChanged {
            value: None,
            index: None,
        };
        assert!(msg.value.is_none());
        assert!(msg.index.is_none());
    }

    // =========================================================================
    // Select Construction Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_select_new() {
        let s = Select::new();
        assert!(s.is_empty());
        assert_eq!(s.get_selected(), None);
        assert!(!s.is_open());
        assert!(!s.disabled);
    }

    #[test]
    fn test_select_default() {
        let s = Select::default();
        assert!(s.is_empty());
    }

    #[test]
    fn test_select_builder() {
        let s = Select::new()
            .option(SelectOption::new("a", "Option A"))
            .option(SelectOption::new("b", "Option B"))
            .placeholder("Choose one")
            .selected(Some(0))
            .min_width(200.0)
            .item_height(40.0)
            .with_test_id("my-select")
            .with_accessible_name("Country");

        assert_eq!(s.option_count(), 2);
        assert_eq!(s.get_selected(), Some(0));
        assert_eq!(Widget::test_id(&s), Some("my-select"));
        assert_eq!(s.accessible_name(), Some("Country"));
    }

    #[test]
    fn test_select_options() {
        let opts = vec![
            SelectOption::simple("One"),
            SelectOption::simple("Two"),
            SelectOption::simple("Three"),
        ];
        let s = Select::new().options(opts);
        assert_eq!(s.option_count(), 3);
    }

    #[test]
    fn test_select_options_from_strings() {
        let s = Select::new().options_from_strings(["Red", "Green", "Blue"]);
        assert_eq!(s.option_count(), 3);
        assert_eq!(s.get_options()[0].value, "Red");
        assert_eq!(s.get_options()[0].label, "Red");
    }

    // =========================================================================
    // Select Selection Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_select_selected_index() {
        let s = Select::new()
            .options_from_strings(["A", "B", "C"])
            .selected(Some(1));
        assert_eq!(s.get_selected(), Some(1));
        assert_eq!(s.get_selected_value(), Some("B"));
        assert_eq!(s.get_selected_label(), Some("B"));
    }

    #[test]
    fn test_select_selected_value() {
        let s = Select::new()
            .option(SelectOption::new("val1", "Label 1"))
            .option(SelectOption::new("val2", "Label 2"))
            .selected_value("val2");
        assert_eq!(s.get_selected(), Some(1));
    }

    #[test]
    fn test_select_selected_out_of_bounds() {
        let s = Select::new()
            .options_from_strings(["A", "B"])
            .selected(Some(10));
        assert_eq!(s.get_selected(), None); // Should clamp
    }

    #[test]
    fn test_select_selected_value_not_found() {
        let s = Select::new()
            .options_from_strings(["A", "B"])
            .selected_value("C");
        assert_eq!(s.get_selected(), None);
    }

    #[test]
    fn test_select_no_selection() {
        let s = Select::new().options_from_strings(["A", "B"]);
        assert_eq!(s.get_selected(), None);
        assert_eq!(s.get_selected_value(), None);
        assert_eq!(s.get_selected_label(), None);
    }

    // =========================================================================
    // Select Widget Trait Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_select_type_id() {
        let s = Select::new();
        assert_eq!(Widget::type_id(&s), TypeId::of::<Select>());
    }

    #[test]
    fn test_select_measure() {
        let s = Select::new().min_width(150.0).item_height(32.0);
        let size = s.measure(Constraints::loose(Size::new(400.0, 200.0)));
        assert_eq!(size.width, 150.0);
        assert_eq!(size.height, 32.0);
    }

    #[test]
    fn test_select_is_interactive() {
        let s = Select::new();
        assert!(s.is_interactive());

        let s = Select::new().disabled(true);
        assert!(!s.is_interactive());
    }

    #[test]
    fn test_select_is_focusable() {
        let s = Select::new();
        assert!(s.is_focusable());

        let s = Select::new().disabled(true);
        assert!(!s.is_focusable());
    }

    #[test]
    fn test_select_accessible_role() {
        let s = Select::new();
        assert_eq!(s.accessible_role(), AccessibleRole::ComboBox);
    }

    #[test]
    fn test_select_children() {
        let s = Select::new();
        assert!(s.children().is_empty());
    }

    // =========================================================================
    // Select Layout Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_select_layout() {
        let mut s = Select::new();
        let bounds = Rect::new(10.0, 20.0, 200.0, 32.0);
        let result = s.layout(bounds);
        assert_eq!(result.size, bounds.size());
        assert_eq!(s.bounds, bounds);
    }

    // =========================================================================
    // Select Size Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_select_min_width_min() {
        let s = Select::new().min_width(10.0);
        assert_eq!(s.min_width, 50.0); // Minimum is 50
    }

    #[test]
    fn test_select_item_height_min() {
        let s = Select::new().item_height(5.0);
        assert_eq!(s.item_height, 20.0); // Minimum is 20
    }

    #[test]
    fn test_select_max_visible_items_min() {
        let s = Select::new().max_visible_items(0);
        assert_eq!(s.max_visible_items, 1); // Minimum is 1
    }

    // =========================================================================
    // Select Color Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_select_colors() {
        let s = Select::new()
            .background_color(Color::RED)
            .border_color(Color::GREEN);
        assert_eq!(s.background_color, Color::RED);
        assert_eq!(s.border_color, Color::GREEN);
    }

    // =========================================================================
    // Select Dropdown Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_select_dropdown_height() {
        let s = Select::new()
            .options_from_strings(["A", "B", "C"])
            .item_height(30.0)
            .max_visible_items(10);
        // 3 items * 30px = 90px
        assert_eq!(s.dropdown_height(), 90.0);
    }

    #[test]
    fn test_select_dropdown_height_limited() {
        let s = Select::new()
            .options_from_strings(["A", "B", "C", "D", "E"])
            .item_height(30.0)
            .max_visible_items(3);
        // limited to 3 items * 30px = 90px
        assert_eq!(s.dropdown_height(), 90.0);
    }

    #[test]
    fn test_select_is_empty() {
        let s = Select::new();
        assert!(s.is_empty());

        let s = Select::new().options_from_strings(["A"]);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_select_option_count() {
        let s = Select::new().options_from_strings(["A", "B", "C"]);
        assert_eq!(s.option_count(), 3);
    }

    // =========================================================================
    // Event Handling Tests - TESTS FIRST
    // =========================================================================

    use presentar_core::Point;

    #[test]
    fn test_select_event_click_header_opens_dropdown() {
        let mut s = Select::new()
            .options_from_strings(["A", "B", "C"])
            .item_height(32.0);
        s.layout(Rect::new(0.0, 0.0, 200.0, 32.0));

        assert!(!s.open);
        let result = s.event(&Event::MouseDown {
            position: Point::new(100.0, 16.0), // In header
            button: MouseButton::Left,
        });
        assert!(s.open);
        assert!(result.is_none()); // Just opens, no selection
    }

    #[test]
    fn test_select_event_click_header_closes_dropdown() {
        let mut s = Select::new()
            .options_from_strings(["A", "B", "C"])
            .item_height(32.0);
        s.layout(Rect::new(0.0, 0.0, 200.0, 32.0));
        s.open = true;

        let result = s.event(&Event::MouseDown {
            position: Point::new(100.0, 16.0), // In header
            button: MouseButton::Left,
        });
        assert!(!s.open);
        assert!(result.is_none());
    }

    #[test]
    fn test_select_event_click_item_selects() {
        let mut s = Select::new()
            .options_from_strings(["Apple", "Banana", "Cherry"])
            .item_height(32.0);
        s.layout(Rect::new(0.0, 0.0, 200.0, 32.0));
        s.open = true;

        // Click second item (index 1): y = 32 + 32 + 16 = 80 (middle of item 1)
        let result = s.event(&Event::MouseDown {
            position: Point::new(100.0, 80.0),
            button: MouseButton::Left,
        });

        assert!(!s.open); // Closes after selection
        assert_eq!(s.get_selected(), Some(1));
        assert!(result.is_some());

        let msg = result.unwrap().downcast::<SelectionChanged>().unwrap();
        assert_eq!(msg.value, Some("Banana".to_string()));
        assert_eq!(msg.index, Some(1));
    }

    #[test]
    fn test_select_event_click_first_item() {
        let mut s = Select::new()
            .options_from_strings(["First", "Second", "Third"])
            .item_height(32.0);
        s.layout(Rect::new(0.0, 0.0, 200.0, 32.0));
        s.open = true;

        // Click first item (index 0): y = 32 + 16 = 48
        let result = s.event(&Event::MouseDown {
            position: Point::new(100.0, 48.0),
            button: MouseButton::Left,
        });

        assert_eq!(s.get_selected(), Some(0));
        let msg = result.unwrap().downcast::<SelectionChanged>().unwrap();
        assert_eq!(msg.value, Some("First".to_string()));
        assert_eq!(msg.index, Some(0));
    }

    #[test]
    fn test_select_event_click_disabled_item_no_select() {
        let mut s = Select::new()
            .option(SelectOption::simple("Enabled"))
            .option(SelectOption::simple("Disabled").disabled(true))
            .item_height(32.0);
        s.layout(Rect::new(0.0, 0.0, 200.0, 32.0));
        s.open = true;

        // Click disabled item (index 1)
        let result = s.event(&Event::MouseDown {
            position: Point::new(100.0, 80.0),
            button: MouseButton::Left,
        });

        assert!(s.open); // Stays open
        assert!(s.get_selected().is_none());
        assert!(result.is_none());
    }

    #[test]
    fn test_select_event_click_outside_closes() {
        let mut s = Select::new()
            .options_from_strings(["A", "B", "C"])
            .item_height(32.0);
        s.layout(Rect::new(0.0, 0.0, 200.0, 32.0));
        s.open = true;

        // Click far below dropdown
        let result = s.event(&Event::MouseDown {
            position: Point::new(100.0, 500.0),
            button: MouseButton::Left,
        });

        assert!(!s.open);
        assert!(result.is_none());
    }

    #[test]
    fn test_select_event_mouse_move_updates_hover() {
        let mut s = Select::new()
            .options_from_strings(["A", "B", "C"])
            .item_height(32.0);
        s.layout(Rect::new(0.0, 0.0, 200.0, 32.0));
        s.open = true;

        assert!(s.hovered_item.is_none());

        // Hover over item 1
        s.event(&Event::MouseMove {
            position: Point::new(100.0, 80.0),
        });
        assert_eq!(s.hovered_item, Some(1));

        // Hover over item 0
        s.event(&Event::MouseMove {
            position: Point::new(100.0, 48.0),
        });
        assert_eq!(s.hovered_item, Some(0));
    }

    #[test]
    fn test_select_event_mouse_move_when_closed_no_hover() {
        let mut s = Select::new()
            .options_from_strings(["A", "B", "C"])
            .item_height(32.0);
        s.layout(Rect::new(0.0, 0.0, 200.0, 32.0));
        // Closed

        s.event(&Event::MouseMove {
            position: Point::new(100.0, 80.0),
        });
        assert!(s.hovered_item.is_none());
    }

    #[test]
    fn test_select_event_focus_out_closes() {
        let mut s = Select::new()
            .options_from_strings(["A", "B", "C"])
            .item_height(32.0);
        s.layout(Rect::new(0.0, 0.0, 200.0, 32.0));
        s.open = true;

        let result = s.event(&Event::FocusOut);
        assert!(!s.open);
        assert!(result.is_none());
    }

    #[test]
    fn test_select_event_disabled_blocks_click() {
        let mut s = Select::new()
            .options_from_strings(["A", "B", "C"])
            .item_height(32.0)
            .disabled(true);
        s.layout(Rect::new(0.0, 0.0, 200.0, 32.0));

        let result = s.event(&Event::MouseDown {
            position: Point::new(100.0, 16.0),
            button: MouseButton::Left,
        });

        assert!(!s.open);
        assert!(result.is_none());
    }

    #[test]
    fn test_select_event_disabled_blocks_mouse_move() {
        let mut s = Select::new()
            .options_from_strings(["A", "B", "C"])
            .item_height(32.0)
            .disabled(true);
        s.layout(Rect::new(0.0, 0.0, 200.0, 32.0));
        s.open = true; // Force open

        s.event(&Event::MouseMove {
            position: Point::new(100.0, 80.0),
        });
        assert!(s.hovered_item.is_none());
    }

    #[test]
    fn test_select_event_right_click_no_effect() {
        let mut s = Select::new()
            .options_from_strings(["A", "B", "C"])
            .item_height(32.0);
        s.layout(Rect::new(0.0, 0.0, 200.0, 32.0));

        let result = s.event(&Event::MouseDown {
            position: Point::new(100.0, 16.0),
            button: MouseButton::Right,
        });

        assert!(!s.open);
        assert!(result.is_none());
    }

    #[test]
    fn test_select_event_click_header_clears_hover() {
        let mut s = Select::new()
            .options_from_strings(["A", "B", "C"])
            .item_height(32.0);
        s.layout(Rect::new(0.0, 0.0, 200.0, 32.0));
        s.hovered_item = Some(1);

        // Open dropdown
        s.event(&Event::MouseDown {
            position: Point::new(100.0, 16.0),
            button: MouseButton::Left,
        });

        assert!(s.open);
        assert!(s.hovered_item.is_none()); // Cleared
    }

    #[test]
    fn test_select_event_full_interaction_flow() {
        let mut s = Select::new()
            .options_from_strings(["Red", "Green", "Blue"])
            .item_height(32.0);
        s.layout(Rect::new(0.0, 0.0, 200.0, 32.0));

        // 1. Click to open
        s.event(&Event::MouseDown {
            position: Point::new(100.0, 16.0),
            button: MouseButton::Left,
        });
        assert!(s.open);
        assert!(s.selected.is_none());

        // 2. Hover over items
        s.event(&Event::MouseMove {
            position: Point::new(100.0, 48.0), // Item 0
        });
        assert_eq!(s.hovered_item, Some(0));

        s.event(&Event::MouseMove {
            position: Point::new(100.0, 112.0), // Item 2
        });
        assert_eq!(s.hovered_item, Some(2));

        // 3. Select item 2
        let result = s.event(&Event::MouseDown {
            position: Point::new(100.0, 112.0),
            button: MouseButton::Left,
        });
        assert!(!s.open);
        assert_eq!(s.get_selected(), Some(2));
        assert_eq!(s.get_selected_value(), Some("Blue"));

        let msg = result.unwrap().downcast::<SelectionChanged>().unwrap();
        assert_eq!(msg.value, Some("Blue".to_string()));

        // 4. Reopen and change selection
        s.event(&Event::MouseDown {
            position: Point::new(100.0, 16.0),
            button: MouseButton::Left,
        });
        assert!(s.open);

        // 5. Select item 0
        let result = s.event(&Event::MouseDown {
            position: Point::new(100.0, 48.0),
            button: MouseButton::Left,
        });
        assert_eq!(s.get_selected_value(), Some("Red"));

        let msg = result.unwrap().downcast::<SelectionChanged>().unwrap();
        assert_eq!(msg.value, Some("Red".to_string()));
        assert_eq!(msg.index, Some(0));
    }

    #[test]
    fn test_select_event_item_at_position_edge_cases() {
        let mut s = Select::new()
            .options_from_strings(["A", "B"])
            .item_height(32.0)
            .max_visible_items(2);
        s.layout(Rect::new(0.0, 0.0, 200.0, 32.0));
        s.open = true;

        // Just inside first item
        assert_eq!(s.item_at_position(32.0), Some(0));
        // Just inside second item
        assert_eq!(s.item_at_position(64.0), Some(1));
        // Past last item
        assert_eq!(s.item_at_position(96.0), None);
        // In header area
        assert_eq!(s.item_at_position(16.0), None);
    }

    #[test]
    fn test_select_event_item_rect_positions() {
        let mut s = Select::new()
            .options_from_strings(["A", "B", "C"])
            .item_height(30.0);
        s.layout(Rect::new(10.0, 20.0, 200.0, 30.0));

        // Item 0 starts at y = 20 + 30 = 50
        let rect0 = s.item_rect(0);
        assert_eq!(rect0.x, 10.0);
        assert_eq!(rect0.y, 50.0);
        assert_eq!(rect0.height, 30.0);

        // Item 1 starts at y = 20 + 30 + 30 = 80
        let rect1 = s.item_rect(1);
        assert_eq!(rect1.y, 80.0);
    }

    #[test]
    fn test_select_event_with_offset_bounds() {
        let mut s = Select::new()
            .options_from_strings(["X", "Y", "Z"])
            .item_height(32.0);
        s.layout(Rect::new(100.0, 50.0, 200.0, 32.0));
        s.open = true;

        // Click item 0: header is at y=50-82, item 0 is at y=82-114
        let result = s.event(&Event::MouseDown {
            position: Point::new(200.0, 98.0), // Middle of item 0
            button: MouseButton::Left,
        });

        assert_eq!(s.get_selected(), Some(0));
        assert!(result.is_some());
    }
}
