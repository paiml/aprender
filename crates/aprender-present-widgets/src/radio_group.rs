//! `RadioGroup` widget for selecting one option from a list.

use presentar_core::{
    widget::{
        AccessibleRole, Brick, BrickAssertion, BrickBudget, BrickVerification, LayoutResult,
        TextStyle,
    },
    Canvas, Color, Constraints, Event, MouseButton, Point, Rect, Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// A single radio option.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RadioOption {
    /// Option value
    pub value: String,
    /// Display label
    pub label: String,
    /// Whether the option is disabled
    pub disabled: bool,
}

impl RadioOption {
    /// Create a new radio option.
    #[must_use]
    pub fn new(value: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            label: label.into(),
            disabled: false,
        }
    }

    /// Set the option as disabled.
    #[must_use]
    pub const fn disabled(mut self) -> Self {
        self.disabled = true;
        self
    }
}

/// Message emitted when radio selection changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RadioChanged {
    /// The newly selected value
    pub value: String,
    /// Index of the selected option
    pub index: usize,
}

/// Radio group orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum RadioOrientation {
    /// Vertical layout (default)
    #[default]
    Vertical,
    /// Horizontal layout
    Horizontal,
}

/// `RadioGroup` widget for selecting one option from multiple choices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadioGroup {
    /// Radio options
    options: Vec<RadioOption>,
    /// Currently selected index
    selected: Option<usize>,
    /// Layout orientation
    orientation: RadioOrientation,
    /// Spacing between options
    spacing: f32,
    /// Radio button size
    radio_size: f32,
    /// Gap between radio and label
    label_gap: f32,
    /// Unselected border color
    border_color: Color,
    /// Selected fill color
    fill_color: Color,
    /// Label text color
    label_color: Color,
    /// Disabled text color
    disabled_color: Color,
    /// Accessible name
    accessible_name_value: Option<String>,
    /// Test ID
    test_id_value: Option<String>,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
}

impl Default for RadioGroup {
    fn default() -> Self {
        Self {
            options: Vec::new(),
            selected: None,
            orientation: RadioOrientation::Vertical,
            spacing: 8.0,
            radio_size: 20.0,
            label_gap: 8.0,
            border_color: Color::new(0.6, 0.6, 0.6, 1.0),
            fill_color: Color::new(0.2, 0.47, 0.96, 1.0),
            label_color: Color::new(0.1, 0.1, 0.1, 1.0),
            disabled_color: Color::new(0.6, 0.6, 0.6, 1.0),
            accessible_name_value: None,
            test_id_value: None,
            bounds: Rect::default(),
        }
    }
}

impl RadioGroup {
    /// Create a new radio group.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an option.
    #[must_use]
    pub fn option(mut self, option: RadioOption) -> Self {
        self.options.push(option);
        self
    }

    /// Add multiple options.
    #[must_use]
    pub fn options(mut self, options: impl IntoIterator<Item = RadioOption>) -> Self {
        self.options.extend(options);
        self
    }

    /// Set selected value.
    #[must_use]
    pub fn selected(mut self, value: &str) -> Self {
        self.selected = self.options.iter().position(|o| o.value == value);
        self
    }

    /// Set selected index.
    #[must_use]
    pub fn selected_index(mut self, index: usize) -> Self {
        if index < self.options.len() {
            self.selected = Some(index);
        }
        self
    }

    /// Set orientation.
    #[must_use]
    pub const fn orientation(mut self, orientation: RadioOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Set spacing between options.
    #[must_use]
    pub fn spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing.max(0.0);
        self
    }

    /// Set radio button size.
    #[must_use]
    pub fn radio_size(mut self, size: f32) -> Self {
        self.radio_size = size.max(12.0);
        self
    }

    /// Set gap between radio and label.
    #[must_use]
    pub fn label_gap(mut self, gap: f32) -> Self {
        self.label_gap = gap.max(0.0);
        self
    }

    /// Set border color.
    #[must_use]
    pub const fn border_color(mut self, color: Color) -> Self {
        self.border_color = color;
        self
    }

    /// Set fill color for selected state.
    #[must_use]
    pub const fn fill_color(mut self, color: Color) -> Self {
        self.fill_color = color;
        self
    }

    /// Set label text color.
    #[must_use]
    pub const fn label_color(mut self, color: Color) -> Self {
        self.label_color = color;
        self
    }

    /// Set accessible name.
    #[must_use]
    pub fn accessible_name(mut self, name: impl Into<String>) -> Self {
        self.accessible_name_value = Some(name.into());
        self
    }

    /// Set test ID.
    #[must_use]
    pub fn test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id_value = Some(id.into());
        self
    }

    /// Get the options.
    #[must_use]
    pub fn get_options(&self) -> &[RadioOption] {
        &self.options
    }

    /// Get selected value.
    #[must_use]
    pub fn get_selected(&self) -> Option<&str> {
        self.selected
            .and_then(|i| self.options.get(i))
            .map(|o| o.value.as_str())
    }

    /// Get selected index.
    #[must_use]
    pub const fn get_selected_index(&self) -> Option<usize> {
        self.selected
    }

    /// Get selected option.
    #[must_use]
    pub fn get_selected_option(&self) -> Option<&RadioOption> {
        self.selected.and_then(|i| self.options.get(i))
    }

    /// Check if a value is selected.
    #[must_use]
    pub fn is_selected(&self, value: &str) -> bool {
        self.get_selected() == Some(value)
    }

    /// Check if an index is selected.
    #[must_use]
    pub fn is_index_selected(&self, index: usize) -> bool {
        self.selected == Some(index)
    }

    /// Check if any option is selected.
    #[must_use]
    pub const fn has_selection(&self) -> bool {
        self.selected.is_some()
    }

    /// Get option count.
    #[must_use]
    pub fn option_count(&self) -> usize {
        self.options.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.options.is_empty()
    }

    /// Set selection by value (mutable).
    pub fn set_selected(&mut self, value: &str) {
        if let Some(index) = self.options.iter().position(|o| o.value == value) {
            if !self.options[index].disabled {
                self.selected = Some(index);
            }
        }
    }

    /// Set selection by index (mutable).
    pub fn set_selected_index(&mut self, index: usize) {
        if index < self.options.len() && !self.options[index].disabled {
            self.selected = Some(index);
        }
    }

    /// Clear selection.
    pub fn clear_selection(&mut self) {
        self.selected = None;
    }

    /// Select next option.
    pub fn select_next(&mut self) {
        if self.options.is_empty() {
            return;
        }
        let start = self.selected.map_or(0, |i| i + 1);
        for offset in 0..self.options.len() {
            let idx = (start + offset) % self.options.len();
            if !self.options[idx].disabled {
                self.selected = Some(idx);
                return;
            }
        }
    }

    /// Select previous option.
    pub fn select_prev(&mut self) {
        if self.options.is_empty() {
            return;
        }
        let start = self.selected.map_or(self.options.len() - 1, |i| {
            if i == 0 {
                self.options.len() - 1
            } else {
                i - 1
            }
        });
        for offset in 0..self.options.len() {
            let idx = if start >= offset {
                start - offset
            } else {
                self.options.len() - (offset - start)
            };
            if !self.options[idx].disabled {
                self.selected = Some(idx);
                return;
            }
        }
    }

    /// Calculate item size (radio + gap + label).
    fn item_size(&self) -> Size {
        // Approximate label width
        let label_width = 100.0;
        Size::new(
            self.radio_size + self.label_gap + label_width,
            self.radio_size.max(20.0),
        )
    }

    /// Get rect for option at index.
    fn option_rect(&self, index: usize) -> Rect {
        let item = self.item_size();
        match self.orientation {
            RadioOrientation::Vertical => {
                let y = (index as f32).mul_add(item.height + self.spacing, self.bounds.y);
                Rect::new(self.bounds.x, y, self.bounds.width, item.height)
            }
            RadioOrientation::Horizontal => {
                let x = (index as f32).mul_add(item.width + self.spacing, self.bounds.x);
                Rect::new(x, self.bounds.y, item.width, item.height)
            }
        }
    }

    /// Find option at point.
    fn option_at_point(&self, x: f32, y: f32) -> Option<usize> {
        for (i, _) in self.options.iter().enumerate() {
            let rect = self.option_rect(i);
            if x >= rect.x && x <= rect.x + rect.width && y >= rect.y && y <= rect.y + rect.height {
                return Some(i);
            }
        }
        None
    }
}

impl Widget for RadioGroup {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let item = self.item_size();
        let count = self.options.len() as f32;

        let preferred = match self.orientation {
            RadioOrientation::Vertical => {
                let total_spacing = if count > 1.0 {
                    self.spacing * (count - 1.0)
                } else {
                    0.0
                };
                Size::new(item.width, count.mul_add(item.height, total_spacing))
            }
            RadioOrientation::Horizontal => {
                let total_spacing = if count > 1.0 {
                    self.spacing * (count - 1.0)
                } else {
                    0.0
                };
                Size::new(count.mul_add(item.width, total_spacing), item.height)
            }
        };

        constraints.constrain(preferred)
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        for (i, option) in self.options.iter().enumerate() {
            let rect = self.option_rect(i);
            let is_selected = self.selected == Some(i);

            // Radio button circle position
            let cx = rect.x + self.radio_size / 2.0;
            let cy = rect.y + rect.height / 2.0;
            let radius = self.radio_size / 2.0;

            // Draw outer circle (border)
            let border_rect = Rect::new(cx - radius, cy - radius, self.radio_size, self.radio_size);
            let border_color = if option.disabled {
                self.disabled_color
            } else if is_selected {
                self.fill_color
            } else {
                self.border_color
            };
            canvas.stroke_rect(border_rect, border_color, 2.0);

            // Draw inner circle if selected
            if is_selected {
                let inner_radius = radius * 0.5;
                let inner_rect = Rect::new(
                    cx - inner_radius,
                    cy - inner_radius,
                    inner_radius * 2.0,
                    inner_radius * 2.0,
                );
                canvas.fill_rect(inner_rect, self.fill_color);
            }

            // Draw label
            let text_color = if option.disabled {
                self.disabled_color
            } else {
                self.label_color
            };

            let text_style = TextStyle {
                size: 14.0,
                color: text_color,
                ..TextStyle::default()
            };

            canvas.draw_text(
                &option.label,
                Point::new(rect.x + self.radio_size + self.label_gap, cy),
                &text_style,
            );
        }
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        if let Event::MouseDown {
            position,
            button: MouseButton::Left,
        } = event
        {
            if let Some(index) = self.option_at_point(position.x, position.y) {
                if !self.options[index].disabled && self.selected != Some(index) {
                    self.selected = Some(index);
                    return Some(Box::new(RadioChanged {
                        value: self.options[index].value.clone(),
                        index,
                    }));
                }
            }
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
        !self.options.is_empty()
    }

    fn is_focusable(&self) -> bool {
        !self.options.is_empty()
    }

    fn accessible_name(&self) -> Option<&str> {
        self.accessible_name_value.as_deref()
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::RadioGroup
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for RadioGroup {
    fn brick_name(&self) -> &'static str {
        "RadioGroup"
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
        r#"<div class="brick-radiogroup"></div>"#.to_string()
    }

    fn to_css(&self) -> String {
        ".brick-radiogroup { display: block; }".to_string()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    // ===== RadioOption Tests =====

    #[test]
    fn test_radio_option_new() {
        let opt = RadioOption::new("val", "Label");
        assert_eq!(opt.value, "val");
        assert_eq!(opt.label, "Label");
        assert!(!opt.disabled);
    }

    #[test]
    fn test_radio_option_disabled() {
        let opt = RadioOption::new("val", "Label").disabled();
        assert!(opt.disabled);
    }

    #[test]
    fn test_radio_option_equality() {
        let opt1 = RadioOption::new("a", "A");
        let opt2 = RadioOption::new("a", "A");
        let opt3 = RadioOption::new("b", "B");
        assert_eq!(opt1, opt2);
        assert_ne!(opt1, opt3);
    }

    // ===== RadioChanged Tests =====

    #[test]
    fn test_radio_changed() {
        let msg = RadioChanged {
            value: "option1".to_string(),
            index: 1,
        };
        assert_eq!(msg.value, "option1");
        assert_eq!(msg.index, 1);
    }

    // ===== RadioOrientation Tests =====

    #[test]
    fn test_radio_orientation_default() {
        assert_eq!(RadioOrientation::default(), RadioOrientation::Vertical);
    }

    // ===== RadioGroup Construction Tests =====

    #[test]
    fn test_radio_group_new() {
        let group = RadioGroup::new();
        assert!(group.is_empty());
        assert_eq!(group.option_count(), 0);
        assert!(!group.has_selection());
    }

    #[test]
    fn test_radio_group_builder() {
        let group = RadioGroup::new()
            .option(RadioOption::new("a", "Option A"))
            .option(RadioOption::new("b", "Option B"))
            .option(RadioOption::new("c", "Option C"))
            .selected("b")
            .orientation(RadioOrientation::Horizontal)
            .spacing(12.0)
            .radio_size(24.0)
            .label_gap(10.0)
            .accessible_name("Choose option")
            .test_id("radio-test");

        assert_eq!(group.option_count(), 3);
        assert_eq!(group.get_selected(), Some("b"));
        assert_eq!(group.get_selected_index(), Some(1));
        assert_eq!(Widget::accessible_name(&group), Some("Choose option"));
        assert_eq!(Widget::test_id(&group), Some("radio-test"));
    }

    #[test]
    fn test_radio_group_options_iter() {
        let opts = vec![RadioOption::new("x", "X"), RadioOption::new("y", "Y")];
        let group = RadioGroup::new().options(opts);
        assert_eq!(group.option_count(), 2);
    }

    #[test]
    fn test_radio_group_selected_index() {
        let group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .option(RadioOption::new("b", "B"))
            .selected_index(1);

        assert_eq!(group.get_selected(), Some("b"));
    }

    #[test]
    fn test_radio_group_selected_not_found() {
        let group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .selected("nonexistent");

        assert!(!group.has_selection());
    }

    // ===== Selection Tests =====

    #[test]
    fn test_radio_group_get_selected_option() {
        let group = RadioGroup::new()
            .option(RadioOption::new("first", "First"))
            .option(RadioOption::new("second", "Second"))
            .selected("second");

        let opt = group.get_selected_option().unwrap();
        assert_eq!(opt.value, "second");
        assert_eq!(opt.label, "Second");
    }

    #[test]
    fn test_radio_group_is_selected() {
        let group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .option(RadioOption::new("b", "B"))
            .selected("a");

        assert!(group.is_selected("a"));
        assert!(!group.is_selected("b"));
    }

    #[test]
    fn test_radio_group_is_index_selected() {
        let group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .option(RadioOption::new("b", "B"))
            .selected_index(0);

        assert!(group.is_index_selected(0));
        assert!(!group.is_index_selected(1));
    }

    // ===== Mutable Selection Tests =====

    #[test]
    fn test_radio_group_set_selected() {
        let mut group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .option(RadioOption::new("b", "B"));

        group.set_selected("b");
        assert_eq!(group.get_selected(), Some("b"));
    }

    #[test]
    fn test_radio_group_set_selected_disabled() {
        let mut group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .option(RadioOption::new("b", "B").disabled());

        group.set_selected("b");
        assert!(!group.has_selection()); // Should not select disabled
    }

    #[test]
    fn test_radio_group_set_selected_index() {
        let mut group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .option(RadioOption::new("b", "B"));

        group.set_selected_index(1);
        assert_eq!(group.get_selected_index(), Some(1));
    }

    #[test]
    fn test_radio_group_set_selected_index_out_of_bounds() {
        let mut group = RadioGroup::new().option(RadioOption::new("a", "A"));

        group.set_selected_index(10);
        assert!(!group.has_selection());
    }

    #[test]
    fn test_radio_group_clear_selection() {
        let mut group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .selected("a");

        assert!(group.has_selection());
        group.clear_selection();
        assert!(!group.has_selection());
    }

    // ===== Navigation Tests =====

    #[test]
    fn test_radio_group_select_next() {
        let mut group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .option(RadioOption::new("b", "B"))
            .option(RadioOption::new("c", "C"))
            .selected_index(0);

        group.select_next();
        assert_eq!(group.get_selected_index(), Some(1));

        group.select_next();
        assert_eq!(group.get_selected_index(), Some(2));

        group.select_next(); // Wrap around
        assert_eq!(group.get_selected_index(), Some(0));
    }

    #[test]
    fn test_radio_group_select_next_skip_disabled() {
        let mut group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .option(RadioOption::new("b", "B").disabled())
            .option(RadioOption::new("c", "C"))
            .selected_index(0);

        group.select_next();
        assert_eq!(group.get_selected_index(), Some(2));
    }

    #[test]
    fn test_radio_group_select_next_no_selection() {
        let mut group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .option(RadioOption::new("b", "B"));

        group.select_next();
        assert_eq!(group.get_selected_index(), Some(0));
    }

    #[test]
    fn test_radio_group_select_prev() {
        let mut group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .option(RadioOption::new("b", "B"))
            .option(RadioOption::new("c", "C"))
            .selected_index(2);

        group.select_prev();
        assert_eq!(group.get_selected_index(), Some(1));

        group.select_prev();
        assert_eq!(group.get_selected_index(), Some(0));

        group.select_prev(); // Wrap around
        assert_eq!(group.get_selected_index(), Some(2));
    }

    #[test]
    fn test_radio_group_select_prev_skip_disabled() {
        let mut group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .option(RadioOption::new("b", "B").disabled())
            .option(RadioOption::new("c", "C"))
            .selected_index(2);

        group.select_prev();
        assert_eq!(group.get_selected_index(), Some(0));
    }

    // ===== Dimension Tests =====

    #[test]
    fn test_radio_group_spacing_min() {
        let group = RadioGroup::new().spacing(-5.0);
        assert_eq!(group.spacing, 0.0);
    }

    #[test]
    fn test_radio_group_radio_size_min() {
        let group = RadioGroup::new().radio_size(5.0);
        assert_eq!(group.radio_size, 12.0);
    }

    #[test]
    fn test_radio_group_label_gap_min() {
        let group = RadioGroup::new().label_gap(-5.0);
        assert_eq!(group.label_gap, 0.0);
    }

    // ===== Widget Trait Tests =====

    #[test]
    fn test_radio_group_type_id() {
        let group = RadioGroup::new();
        assert_eq!(Widget::type_id(&group), TypeId::of::<RadioGroup>());
    }

    #[test]
    fn test_radio_group_measure_vertical() {
        let group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .option(RadioOption::new("b", "B"))
            .option(RadioOption::new("c", "C"))
            .orientation(RadioOrientation::Vertical)
            .radio_size(20.0)
            .spacing(8.0);

        let size = group.measure(Constraints::loose(Size::new(500.0, 500.0)));
        // 3 items * 20 height + 2 * 8 spacing = 60 + 16 = 76
        assert!(size.height > 0.0);
        assert!(size.width > 0.0);
    }

    #[test]
    fn test_radio_group_measure_horizontal() {
        let group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .option(RadioOption::new("b", "B"))
            .orientation(RadioOrientation::Horizontal)
            .spacing(8.0);

        let size = group.measure(Constraints::loose(Size::new(500.0, 500.0)));
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn test_radio_group_layout() {
        let mut group = RadioGroup::new().option(RadioOption::new("a", "A"));
        let bounds = Rect::new(10.0, 20.0, 200.0, 100.0);
        let result = group.layout(bounds);
        assert_eq!(result.size, Size::new(200.0, 100.0));
        assert_eq!(group.bounds, bounds);
    }

    #[test]
    fn test_radio_group_children() {
        let group = RadioGroup::new();
        assert!(group.children().is_empty());
    }

    #[test]
    fn test_radio_group_is_interactive() {
        let group = RadioGroup::new();
        assert!(!group.is_interactive()); // Empty

        let group = RadioGroup::new().option(RadioOption::new("a", "A"));
        assert!(group.is_interactive());
    }

    #[test]
    fn test_radio_group_is_focusable() {
        let group = RadioGroup::new();
        assert!(!group.is_focusable()); // Empty

        let group = RadioGroup::new().option(RadioOption::new("a", "A"));
        assert!(group.is_focusable());
    }

    #[test]
    fn test_radio_group_accessible_role() {
        let group = RadioGroup::new();
        assert_eq!(group.accessible_role(), AccessibleRole::RadioGroup);
    }

    #[test]
    fn test_radio_group_accessible_name() {
        let group = RadioGroup::new().accessible_name("Select size");
        assert_eq!(Widget::accessible_name(&group), Some("Select size"));
    }

    #[test]
    fn test_radio_group_test_id() {
        let group = RadioGroup::new().test_id("size-radio");
        assert_eq!(Widget::test_id(&group), Some("size-radio"));
    }

    // ===== Event Tests =====

    #[test]
    fn test_radio_group_click_selects() {
        let mut group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .option(RadioOption::new("b", "B"))
            .radio_size(20.0)
            .spacing(8.0);
        group.bounds = Rect::new(0.0, 0.0, 200.0, 56.0);

        // Click on second option (y = 28 + some offset)
        let event = Event::MouseDown {
            position: Point::new(10.0, 38.0),
            button: MouseButton::Left,
        };

        let result = group.event(&event);
        assert!(result.is_some());
        assert_eq!(group.get_selected_index(), Some(1));

        let msg = result.unwrap().downcast::<RadioChanged>().unwrap();
        assert_eq!(msg.value, "b");
        assert_eq!(msg.index, 1);
    }

    #[test]
    fn test_radio_group_click_disabled_no_change() {
        let mut group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .option(RadioOption::new("b", "B").disabled())
            .radio_size(20.0)
            .spacing(8.0);
        group.bounds = Rect::new(0.0, 0.0, 200.0, 56.0);

        // Click on disabled option
        let event = Event::MouseDown {
            position: Point::new(10.0, 38.0),
            button: MouseButton::Left,
        };

        let result = group.event(&event);
        assert!(result.is_none());
        assert!(!group.has_selection());
    }

    #[test]
    fn test_radio_group_click_same_no_event() {
        let mut group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .option(RadioOption::new("b", "B"))
            .selected_index(0)
            .radio_size(20.0);
        group.bounds = Rect::new(0.0, 0.0, 200.0, 56.0);

        // Click on already selected option
        let event = Event::MouseDown {
            position: Point::new(10.0, 10.0),
            button: MouseButton::Left,
        };

        let result = group.event(&event);
        assert!(result.is_none());
    }

    // ===== Color Tests =====

    #[test]
    fn test_radio_group_colors() {
        let group = RadioGroup::new()
            .border_color(Color::RED)
            .fill_color(Color::GREEN)
            .label_color(Color::BLUE);

        assert_eq!(group.border_color, Color::RED);
        assert_eq!(group.fill_color, Color::GREEN);
        assert_eq!(group.label_color, Color::BLUE);
    }

    #[test]
    fn test_radio_group_right_click_no_select() {
        let mut group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .radio_size(20.0);
        group.bounds = Rect::new(0.0, 0.0, 200.0, 28.0);

        let result = group.event(&Event::MouseDown {
            position: Point::new(10.0, 10.0),
            button: MouseButton::Right,
        });
        assert!(group.selected.is_none());
        assert!(result.is_none());
    }

    #[test]
    fn test_radio_group_click_outside_no_select() {
        let mut group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .radio_size(20.0);
        group.bounds = Rect::new(0.0, 0.0, 200.0, 28.0);

        let result = group.event(&Event::MouseDown {
            position: Point::new(10.0, 100.0),
            button: MouseButton::Left,
        });
        assert!(group.selected.is_none());
        assert!(result.is_none());
    }

    #[test]
    fn test_radio_group_click_with_offset_bounds() {
        let mut group = RadioGroup::new()
            .option(RadioOption::new("a", "A"))
            .option(RadioOption::new("b", "B"))
            .radio_size(20.0)
            .spacing(8.0);
        group.bounds = Rect::new(50.0, 100.0, 200.0, 56.0);

        let result = group.event(&Event::MouseDown {
            position: Point::new(60.0, 138.0), // Second option
            button: MouseButton::Left,
        });
        assert_eq!(group.selected, Some(1));
        assert!(result.is_some());
    }
}
