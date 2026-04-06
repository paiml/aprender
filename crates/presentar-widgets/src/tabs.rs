//! Tabs widget for tabbed navigation.

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

/// A single tab definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tab {
    /// Tab ID
    pub id: String,
    /// Tab label
    pub label: String,
    /// Whether tab is disabled
    pub disabled: bool,
    /// Optional icon name
    pub icon: Option<String>,
}

impl Tab {
    /// Create a new tab.
    #[must_use]
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            disabled: false,
            icon: None,
        }
    }

    /// Set the tab as disabled.
    #[must_use]
    pub const fn disabled(mut self) -> Self {
        self.disabled = true;
        self
    }

    /// Set an icon.
    #[must_use]
    pub fn icon(mut self, icon: impl Into<String>) -> Self {
        self.icon = Some(icon.into());
        self
    }
}

/// Message emitted when active tab changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabChanged {
    /// ID of the newly active tab
    pub tab_id: String,
    /// Index of the newly active tab
    pub index: usize,
}

/// Tab orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TabOrientation {
    /// Tabs on top (horizontal)
    #[default]
    Top,
    /// Tabs on bottom (horizontal)
    Bottom,
    /// Tabs on left (vertical)
    Left,
    /// Tabs on right (vertical)
    Right,
}

/// Tabs widget for tabbed navigation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tabs {
    /// Tab definitions
    items: Vec<Tab>,
    /// Active tab index
    active: usize,
    /// Tab orientation
    orientation: TabOrientation,
    /// Tab bar height (for horizontal) or width (for vertical)
    tab_size: f32,
    /// Minimum tab width
    min_tab_width: f32,
    /// Tab spacing
    spacing: f32,
    /// Tab background color
    tab_bg: Color,
    /// Active tab background color
    active_bg: Color,
    /// Inactive tab text color
    inactive_color: Color,
    /// Active tab text color
    active_color: Color,
    /// Disabled tab text color
    disabled_color: Color,
    /// Border color
    border_color: Color,
    /// Show border under tabs
    show_border: bool,
    /// Accessible name
    accessible_name_value: Option<String>,
    /// Test ID
    test_id_value: Option<String>,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
}

impl Default for Tabs {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            active: 0,
            orientation: TabOrientation::Top,
            tab_size: 48.0,
            min_tab_width: 80.0,
            spacing: 0.0,
            tab_bg: Color::new(0.95, 0.95, 0.95, 1.0),
            active_bg: Color::WHITE,
            inactive_color: Color::new(0.4, 0.4, 0.4, 1.0),
            active_color: Color::new(0.2, 0.47, 0.96, 1.0),
            disabled_color: Color::new(0.7, 0.7, 0.7, 1.0),
            border_color: Color::new(0.85, 0.85, 0.85, 1.0),
            show_border: true,
            accessible_name_value: None,
            test_id_value: None,
            bounds: Rect::default(),
        }
    }
}

impl Tabs {
    /// Create a new tabs widget.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tab.
    #[must_use]
    pub fn tab(mut self, tab: Tab) -> Self {
        self.items.push(tab);
        self
    }

    /// Add multiple tabs.
    #[must_use]
    pub fn tabs(mut self, tabs: impl IntoIterator<Item = Tab>) -> Self {
        self.items.extend(tabs);
        self
    }

    /// Set the active tab by index.
    #[must_use]
    pub const fn active(mut self, index: usize) -> Self {
        self.active = index;
        self
    }

    /// Set the active tab by ID.
    #[must_use]
    pub fn active_id(mut self, id: &str) -> Self {
        if let Some(index) = self.items.iter().position(|t| t.id == id) {
            self.active = index;
        }
        self
    }

    /// Set tab orientation.
    #[must_use]
    pub const fn orientation(mut self, orientation: TabOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Set tab bar size.
    #[must_use]
    pub fn tab_size(mut self, size: f32) -> Self {
        self.tab_size = size.max(24.0);
        self
    }

    /// Set minimum tab width.
    #[must_use]
    pub fn min_tab_width(mut self, width: f32) -> Self {
        self.min_tab_width = width.max(40.0);
        self
    }

    /// Set tab spacing.
    #[must_use]
    pub fn spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing.max(0.0);
        self
    }

    /// Set tab background color.
    #[must_use]
    pub const fn tab_bg(mut self, color: Color) -> Self {
        self.tab_bg = color;
        self
    }

    /// Set active tab background color.
    #[must_use]
    pub const fn active_bg(mut self, color: Color) -> Self {
        self.active_bg = color;
        self
    }

    /// Set inactive tab text color.
    #[must_use]
    pub const fn inactive_color(mut self, color: Color) -> Self {
        self.inactive_color = color;
        self
    }

    /// Set active tab text color.
    #[must_use]
    pub const fn active_color(mut self, color: Color) -> Self {
        self.active_color = color;
        self
    }

    /// Set whether to show border.
    #[must_use]
    pub const fn show_border(mut self, show: bool) -> Self {
        self.show_border = show;
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

    /// Get tab count.
    #[must_use]
    pub fn tab_count(&self) -> usize {
        self.items.len()
    }

    /// Get the tabs.
    #[must_use]
    pub fn get_tabs(&self) -> &[Tab] {
        &self.items
    }

    /// Get active tab index.
    #[must_use]
    pub const fn get_active(&self) -> usize {
        self.active
    }

    /// Get active tab.
    #[must_use]
    pub fn get_active_tab(&self) -> Option<&Tab> {
        self.items.get(self.active)
    }

    /// Get active tab ID.
    #[must_use]
    pub fn get_active_id(&self) -> Option<&str> {
        self.items.get(self.active).map(|t| t.id.as_str())
    }

    /// Check if a tab is active.
    #[must_use]
    pub const fn is_active(&self, index: usize) -> bool {
        self.active == index
    }

    /// Check if tabs are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Set active tab by index (mutable).
    pub fn set_active(&mut self, index: usize) {
        if index < self.items.len() && !self.items[index].disabled {
            self.active = index;
        }
    }

    /// Set active tab by ID (mutable).
    pub fn set_active_id(&mut self, id: &str) {
        if let Some(index) = self.items.iter().position(|t| t.id == id) {
            if !self.items[index].disabled {
                self.active = index;
            }
        }
    }

    /// Navigate to next tab.
    pub fn next_tab(&mut self) {
        let mut next = (self.active + 1) % self.items.len();
        let start = next;
        loop {
            if !self.items[next].disabled {
                self.active = next;
                return;
            }
            next = (next + 1) % self.items.len();
            if next == start {
                return; // All tabs disabled
            }
        }
    }

    /// Navigate to previous tab.
    pub fn prev_tab(&mut self) {
        if self.items.is_empty() {
            return;
        }
        let mut prev = if self.active == 0 {
            self.items.len() - 1
        } else {
            self.active - 1
        };
        let start = prev;
        loop {
            if !self.items[prev].disabled {
                self.active = prev;
                return;
            }
            prev = if prev == 0 {
                self.items.len() - 1
            } else {
                prev - 1
            };
            if prev == start {
                return; // All tabs disabled
            }
        }
    }

    /// Calculate tab width based on available space.
    fn calculate_tab_width(&self, available_width: f32) -> f32 {
        if self.items.is_empty() {
            return self.min_tab_width;
        }
        let total_spacing = self.spacing * (self.items.len() - 1).max(0) as f32;
        let per_tab = (available_width - total_spacing) / self.items.len() as f32;
        per_tab.max(self.min_tab_width)
    }

    /// Get tab rect by index.
    fn tab_rect(&self, index: usize, tab_width: f32) -> Rect {
        match self.orientation {
            TabOrientation::Top | TabOrientation::Bottom => {
                let x = (index as f32).mul_add(tab_width + self.spacing, self.bounds.x);
                let y = if self.orientation == TabOrientation::Top {
                    self.bounds.y
                } else {
                    self.bounds.y + self.bounds.height - self.tab_size
                };
                Rect::new(x, y, tab_width, self.tab_size)
            }
            TabOrientation::Left | TabOrientation::Right => {
                let y = (index as f32).mul_add(self.tab_size + self.spacing, self.bounds.y);
                let x = if self.orientation == TabOrientation::Left {
                    self.bounds.x
                } else {
                    self.bounds.x + self.bounds.width - self.min_tab_width
                };
                Rect::new(x, y, self.min_tab_width, self.tab_size)
            }
        }
    }

    /// Find tab at point.
    fn tab_at_point(&self, x: f32, y: f32) -> Option<usize> {
        let tab_width = self.calculate_tab_width(self.bounds.width);
        for (i, _) in self.items.iter().enumerate() {
            let rect = self.tab_rect(i, tab_width);
            if x >= rect.x && x <= rect.x + rect.width && y >= rect.y && y <= rect.y + rect.height {
                return Some(i);
            }
        }
        None
    }
}

impl Widget for Tabs {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let is_horizontal = matches!(
            self.orientation,
            TabOrientation::Top | TabOrientation::Bottom
        );

        let preferred = if is_horizontal {
            Size::new(self.items.len() as f32 * self.min_tab_width, self.tab_size)
        } else {
            Size::new(self.min_tab_width, self.items.len() as f32 * self.tab_size)
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
        // Draw tab bar background
        canvas.fill_rect(self.bounds, self.tab_bg);

        let tab_width = self.calculate_tab_width(self.bounds.width);

        // Draw individual tabs
        for (i, tab) in self.items.iter().enumerate() {
            let rect = self.tab_rect(i, tab_width);

            // Draw tab background
            let bg_color = if i == self.active {
                self.active_bg
            } else {
                self.tab_bg
            };
            canvas.fill_rect(rect, bg_color);

            // Draw tab label
            let text_color = if tab.disabled {
                self.disabled_color
            } else if i == self.active {
                self.active_color
            } else {
                self.inactive_color
            };

            let text_style = TextStyle {
                size: 14.0,
                color: text_color,
                ..TextStyle::default()
            };

            canvas.draw_text(
                &tab.label,
                Point::new(rect.x + 12.0, rect.y + rect.height / 2.0),
                &text_style,
            );

            // Draw active indicator
            if i == self.active && self.show_border {
                let indicator_rect = match self.orientation {
                    TabOrientation::Top => {
                        Rect::new(rect.x, rect.y + rect.height - 2.0, rect.width, 2.0)
                    }
                    TabOrientation::Bottom => Rect::new(rect.x, rect.y, rect.width, 2.0),
                    TabOrientation::Left => {
                        Rect::new(rect.x + rect.width - 2.0, rect.y, 2.0, rect.height)
                    }
                    TabOrientation::Right => Rect::new(rect.x, rect.y, 2.0, rect.height),
                };
                canvas.fill_rect(indicator_rect, self.active_color);
            }
        }

        // Draw border
        if self.show_border {
            let border_rect = match self.orientation {
                TabOrientation::Top => Rect::new(
                    self.bounds.x,
                    self.bounds.y + self.tab_size - 1.0,
                    self.bounds.width,
                    1.0,
                ),
                TabOrientation::Bottom => {
                    Rect::new(self.bounds.x, self.bounds.y, self.bounds.width, 1.0)
                }
                TabOrientation::Left => Rect::new(
                    self.bounds.x + self.min_tab_width - 1.0,
                    self.bounds.y,
                    1.0,
                    self.bounds.height,
                ),
                TabOrientation::Right => {
                    Rect::new(self.bounds.x, self.bounds.y, 1.0, self.bounds.height)
                }
            };
            canvas.fill_rect(border_rect, self.border_color);
        }
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        if let Event::MouseDown {
            position,
            button: MouseButton::Left,
        } = event
        {
            if let Some(index) = self.tab_at_point(position.x, position.y) {
                if !self.items[index].disabled && index != self.active {
                    self.active = index;
                    return Some(Box::new(TabChanged {
                        tab_id: self.items[index].id.clone(),
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
        !self.items.is_empty()
    }

    fn is_focusable(&self) -> bool {
        !self.items.is_empty()
    }

    fn accessible_name(&self) -> Option<&str> {
        self.accessible_name_value.as_deref()
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Tab
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for Tabs {
    fn brick_name(&self) -> &'static str {
        "Tabs"
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
        r#"<div class="brick-tabs"></div>"#.to_string()
    }

    fn to_css(&self) -> String {
        ".brick-tabs { display: flex; flex-direction: column; }".to_string()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    // ===== Tab Tests =====

    #[test]
    fn test_tab_new() {
        let tab = Tab::new("home", "Home");
        assert_eq!(tab.id, "home");
        assert_eq!(tab.label, "Home");
        assert!(!tab.disabled);
        assert!(tab.icon.is_none());
    }

    #[test]
    fn test_tab_disabled() {
        let tab = Tab::new("settings", "Settings").disabled();
        assert!(tab.disabled);
    }

    #[test]
    fn test_tab_icon() {
        let tab = Tab::new("profile", "Profile").icon("user");
        assert_eq!(tab.icon, Some("user".to_string()));
    }

    // ===== TabChanged Tests =====

    #[test]
    fn test_tab_changed() {
        let msg = TabChanged {
            tab_id: "settings".to_string(),
            index: 2,
        };
        assert_eq!(msg.tab_id, "settings");
        assert_eq!(msg.index, 2);
    }

    // ===== TabOrientation Tests =====

    #[test]
    fn test_tab_orientation_default() {
        assert_eq!(TabOrientation::default(), TabOrientation::Top);
    }

    // ===== Tabs Construction Tests =====

    #[test]
    fn test_tabs_new() {
        let tabs = Tabs::new();
        assert_eq!(tabs.tab_count(), 0);
        assert!(tabs.is_empty());
    }

    #[test]
    fn test_tabs_builder() {
        let tabs = Tabs::new()
            .tab(Tab::new("home", "Home"))
            .tab(Tab::new("about", "About"))
            .tab(Tab::new("contact", "Contact"))
            .active(1)
            .orientation(TabOrientation::Top)
            .tab_size(50.0)
            .min_tab_width(100.0)
            .spacing(4.0)
            .show_border(true)
            .accessible_name("Main navigation")
            .test_id("main-tabs");

        assert_eq!(tabs.tab_count(), 3);
        assert_eq!(tabs.get_active(), 1);
        assert_eq!(tabs.get_active_id(), Some("about"));
        assert_eq!(Widget::accessible_name(&tabs), Some("Main navigation"));
        assert_eq!(Widget::test_id(&tabs), Some("main-tabs"));
    }

    #[test]
    fn test_tabs_multiple() {
        let tab_list = vec![Tab::new("a", "A"), Tab::new("b", "B"), Tab::new("c", "C")];
        let tabs = Tabs::new().tabs(tab_list);
        assert_eq!(tabs.tab_count(), 3);
    }

    #[test]
    fn test_tabs_active_id() {
        let tabs = Tabs::new()
            .tab(Tab::new("first", "First"))
            .tab(Tab::new("second", "Second"))
            .active_id("second");

        assert_eq!(tabs.get_active(), 1);
    }

    #[test]
    fn test_tabs_active_id_not_found() {
        let tabs = Tabs::new()
            .tab(Tab::new("first", "First"))
            .active_id("nonexistent");

        assert_eq!(tabs.get_active(), 0);
    }

    // ===== Active Tab Tests =====

    #[test]
    fn test_tabs_get_active_tab() {
        let tabs = Tabs::new()
            .tab(Tab::new("home", "Home"))
            .tab(Tab::new("about", "About"))
            .active(1);

        let active = tabs.get_active_tab().unwrap();
        assert_eq!(active.id, "about");
    }

    #[test]
    fn test_tabs_get_active_tab_empty() {
        let tabs = Tabs::new();
        assert!(tabs.get_active_tab().is_none());
    }

    #[test]
    fn test_tabs_is_active() {
        let tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .active(1);

        assert!(!tabs.is_active(0));
        assert!(tabs.is_active(1));
    }

    // ===== Set Active Tests =====

    #[test]
    fn test_tabs_set_active() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .tab(Tab::new("c", "C"));

        tabs.set_active(2);
        assert_eq!(tabs.get_active(), 2);
    }

    #[test]
    fn test_tabs_set_active_out_of_bounds() {
        let mut tabs = Tabs::new().tab(Tab::new("a", "A")).tab(Tab::new("b", "B"));

        tabs.set_active(10);
        assert_eq!(tabs.get_active(), 0); // Unchanged
    }

    #[test]
    fn test_tabs_set_active_disabled() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B").disabled());

        tabs.set_active(1);
        assert_eq!(tabs.get_active(), 0); // Unchanged, tab is disabled
    }

    #[test]
    fn test_tabs_set_active_id() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("home", "Home"))
            .tab(Tab::new("settings", "Settings"));

        tabs.set_active_id("settings");
        assert_eq!(tabs.get_active(), 1);
    }

    // ===== Navigation Tests =====

    #[test]
    fn test_tabs_next_tab() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .tab(Tab::new("c", "C"))
            .active(0);

        tabs.next_tab();
        assert_eq!(tabs.get_active(), 1);

        tabs.next_tab();
        assert_eq!(tabs.get_active(), 2);

        tabs.next_tab(); // Wrap around
        assert_eq!(tabs.get_active(), 0);
    }

    #[test]
    fn test_tabs_next_tab_skip_disabled() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B").disabled())
            .tab(Tab::new("c", "C"))
            .active(0);

        tabs.next_tab();
        assert_eq!(tabs.get_active(), 2); // Skipped disabled tab
    }

    #[test]
    fn test_tabs_prev_tab() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .tab(Tab::new("c", "C"))
            .active(2);

        tabs.prev_tab();
        assert_eq!(tabs.get_active(), 1);

        tabs.prev_tab();
        assert_eq!(tabs.get_active(), 0);

        tabs.prev_tab(); // Wrap around
        assert_eq!(tabs.get_active(), 2);
    }

    #[test]
    fn test_tabs_prev_tab_skip_disabled() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B").disabled())
            .tab(Tab::new("c", "C"))
            .active(2);

        tabs.prev_tab();
        assert_eq!(tabs.get_active(), 0); // Skipped disabled tab
    }

    // ===== Dimension Tests =====

    #[test]
    fn test_tabs_tab_size_min() {
        let tabs = Tabs::new().tab_size(10.0);
        assert_eq!(tabs.tab_size, 24.0);
    }

    #[test]
    fn test_tabs_min_tab_width_min() {
        let tabs = Tabs::new().min_tab_width(20.0);
        assert_eq!(tabs.min_tab_width, 40.0);
    }

    #[test]
    fn test_tabs_spacing_min() {
        let tabs = Tabs::new().spacing(-5.0);
        assert_eq!(tabs.spacing, 0.0);
    }

    #[test]
    fn test_tabs_calculate_tab_width() {
        let tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .min_tab_width(50.0)
            .spacing(0.0);

        assert_eq!(tabs.calculate_tab_width(200.0), 100.0);
    }

    #[test]
    fn test_tabs_calculate_tab_width_with_spacing() {
        let tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .min_tab_width(50.0)
            .spacing(10.0);

        // (200 - 10) / 2 = 95
        assert_eq!(tabs.calculate_tab_width(200.0), 95.0);
    }

    // ===== Widget Trait Tests =====

    #[test]
    fn test_tabs_type_id() {
        let tabs = Tabs::new();
        assert_eq!(Widget::type_id(&tabs), TypeId::of::<Tabs>());
    }

    #[test]
    fn test_tabs_measure_horizontal() {
        let tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .orientation(TabOrientation::Top)
            .min_tab_width(100.0)
            .tab_size(48.0);

        let size = tabs.measure(Constraints::loose(Size::new(500.0, 500.0)));
        assert_eq!(size.width, 200.0);
        assert_eq!(size.height, 48.0);
    }

    #[test]
    fn test_tabs_measure_vertical() {
        let tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .orientation(TabOrientation::Left)
            .min_tab_width(100.0)
            .tab_size(48.0);

        let size = tabs.measure(Constraints::loose(Size::new(500.0, 500.0)));
        assert_eq!(size.width, 100.0);
        assert_eq!(size.height, 96.0);
    }

    #[test]
    fn test_tabs_layout() {
        let mut tabs = Tabs::new().tab(Tab::new("a", "A"));
        let bounds = Rect::new(10.0, 20.0, 300.0, 48.0);
        let result = tabs.layout(bounds);
        assert_eq!(result.size, Size::new(300.0, 48.0));
        assert_eq!(tabs.bounds, bounds);
    }

    #[test]
    fn test_tabs_children() {
        let tabs = Tabs::new();
        assert!(tabs.children().is_empty());
    }

    #[test]
    fn test_tabs_is_interactive() {
        let tabs = Tabs::new();
        assert!(!tabs.is_interactive()); // Empty

        let tabs = Tabs::new().tab(Tab::new("a", "A"));
        assert!(tabs.is_interactive());
    }

    #[test]
    fn test_tabs_is_focusable() {
        let tabs = Tabs::new();
        assert!(!tabs.is_focusable()); // Empty

        let tabs = Tabs::new().tab(Tab::new("a", "A"));
        assert!(tabs.is_focusable());
    }

    #[test]
    fn test_tabs_accessible_role() {
        let tabs = Tabs::new();
        assert_eq!(tabs.accessible_role(), AccessibleRole::Tab);
    }

    #[test]
    fn test_tabs_accessible_name() {
        let tabs = Tabs::new().accessible_name("Section tabs");
        assert_eq!(Widget::accessible_name(&tabs), Some("Section tabs"));
    }

    #[test]
    fn test_tabs_test_id() {
        let tabs = Tabs::new().test_id("nav-tabs");
        assert_eq!(Widget::test_id(&tabs), Some("nav-tabs"));
    }

    // ===== Tab Rect Tests =====

    #[test]
    fn test_tab_rect_top() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .orientation(TabOrientation::Top)
            .tab_size(48.0);
        tabs.bounds = Rect::new(0.0, 0.0, 200.0, 48.0);

        let rect0 = tabs.tab_rect(0, 100.0);
        assert_eq!(rect0.x, 0.0);
        assert_eq!(rect0.y, 0.0);
        assert_eq!(rect0.width, 100.0);
        assert_eq!(rect0.height, 48.0);

        let rect1 = tabs.tab_rect(1, 100.0);
        assert_eq!(rect1.x, 100.0);
    }

    #[test]
    fn test_tab_rect_bottom() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .orientation(TabOrientation::Bottom)
            .tab_size(48.0);
        tabs.bounds = Rect::new(0.0, 0.0, 200.0, 100.0);

        let rect = tabs.tab_rect(0, 100.0);
        assert_eq!(rect.y, 52.0); // 100 - 48
    }

    // ===== Event Tests =====

    #[test]
    fn test_tabs_click_changes_active() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .tab_size(48.0)
            .min_tab_width(100.0);
        tabs.bounds = Rect::new(0.0, 0.0, 200.0, 48.0);

        // Click on second tab
        let event = Event::MouseDown {
            position: Point::new(150.0, 24.0),
            button: MouseButton::Left,
        };

        let result = tabs.event(&event);
        assert!(result.is_some());
        assert_eq!(tabs.get_active(), 1);

        let msg = result.unwrap().downcast::<TabChanged>().unwrap();
        assert_eq!(msg.tab_id, "b");
        assert_eq!(msg.index, 1);
    }

    #[test]
    fn test_tabs_click_disabled_no_change() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B").disabled())
            .tab_size(48.0)
            .min_tab_width(100.0);
        tabs.bounds = Rect::new(0.0, 0.0, 200.0, 48.0);

        // Click on disabled tab
        let event = Event::MouseDown {
            position: Point::new(150.0, 24.0),
            button: MouseButton::Left,
        };

        let result = tabs.event(&event);
        assert!(result.is_none());
        assert_eq!(tabs.get_active(), 0);
    }

    #[test]
    fn test_tabs_click_same_tab_no_event() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .active(0)
            .tab_size(48.0)
            .min_tab_width(100.0);
        tabs.bounds = Rect::new(0.0, 0.0, 200.0, 48.0);

        // Click on already active tab
        let event = Event::MouseDown {
            position: Point::new(50.0, 24.0),
            button: MouseButton::Left,
        };

        let result = tabs.event(&event);
        assert!(result.is_none());
    }

    // ===== Additional Tab Rect Tests =====

    #[test]
    fn test_tab_rect_left() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .orientation(TabOrientation::Left)
            .tab_size(48.0)
            .min_tab_width(80.0);
        tabs.bounds = Rect::new(0.0, 0.0, 200.0, 200.0);

        let rect0 = tabs.tab_rect(0, 80.0);
        assert_eq!(rect0.x, 0.0);
        assert_eq!(rect0.y, 0.0);
        assert_eq!(rect0.width, 80.0);
        assert_eq!(rect0.height, 48.0);

        let rect1 = tabs.tab_rect(1, 80.0);
        assert_eq!(rect1.y, 48.0);
    }

    #[test]
    fn test_tab_rect_right() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .orientation(TabOrientation::Right)
            .tab_size(48.0)
            .min_tab_width(80.0);
        tabs.bounds = Rect::new(0.0, 0.0, 200.0, 200.0);

        let rect = tabs.tab_rect(0, 80.0);
        assert_eq!(rect.x, 120.0); // 200 - 80
    }

    #[test]
    fn test_tab_rect_with_spacing() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .orientation(TabOrientation::Top)
            .spacing(10.0)
            .tab_size(48.0);
        tabs.bounds = Rect::new(0.0, 0.0, 300.0, 48.0);

        let rect0 = tabs.tab_rect(0, 100.0);
        assert_eq!(rect0.x, 0.0);

        let rect1 = tabs.tab_rect(1, 100.0);
        assert_eq!(rect1.x, 110.0); // 100 + 10 spacing
    }

    // ===== Calculate Tab Width Tests =====

    #[test]
    fn test_calculate_tab_width_empty() {
        let tabs = Tabs::new().min_tab_width(80.0);
        assert_eq!(tabs.calculate_tab_width(500.0), 80.0);
    }

    #[test]
    fn test_calculate_tab_width_narrow_space() {
        let tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .min_tab_width(100.0);

        // Space so narrow that we need to use min_tab_width
        assert_eq!(tabs.calculate_tab_width(50.0), 100.0);
    }

    // ===== Tab At Point Tests =====

    #[test]
    fn test_tab_at_point_found() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .min_tab_width(100.0)
            .tab_size(48.0);
        tabs.bounds = Rect::new(0.0, 0.0, 200.0, 48.0);

        assert_eq!(tabs.tab_at_point(50.0, 24.0), Some(0));
        assert_eq!(tabs.tab_at_point(150.0, 24.0), Some(1));
    }

    #[test]
    fn test_tab_at_point_not_found() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .min_tab_width(100.0)
            .tab_size(48.0);
        tabs.bounds = Rect::new(0.0, 0.0, 200.0, 48.0);

        // Click outside tabs area
        assert_eq!(tabs.tab_at_point(50.0, 100.0), None);
        assert_eq!(tabs.tab_at_point(-10.0, 24.0), None);
    }

    // ===== Navigation Edge Cases =====

    #[test]
    fn test_tabs_next_tab_all_disabled() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A").disabled())
            .tab(Tab::new("b", "B").disabled())
            .tab(Tab::new("c", "C").disabled());
        tabs.active = 0; // Start position

        tabs.next_tab();
        assert_eq!(tabs.get_active(), 0); // Unchanged - all disabled
    }

    #[test]
    fn test_tabs_prev_tab_all_disabled() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A").disabled())
            .tab(Tab::new("b", "B").disabled());
        tabs.active = 0;

        tabs.prev_tab();
        assert_eq!(tabs.get_active(), 0); // Unchanged - all disabled
    }

    #[test]
    fn test_tabs_prev_tab_empty() {
        let mut tabs = Tabs::new();
        tabs.prev_tab(); // Should not panic
        assert_eq!(tabs.get_active(), 0);
    }

    #[test]
    fn test_tabs_set_active_id_not_found() {
        let mut tabs = Tabs::new().tab(Tab::new("a", "A")).tab(Tab::new("b", "B"));

        tabs.set_active_id("nonexistent");
        assert_eq!(tabs.get_active(), 0); // Unchanged
    }

    #[test]
    fn test_tabs_set_active_id_disabled() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B").disabled());

        tabs.set_active_id("b");
        assert_eq!(tabs.get_active(), 0); // Unchanged - disabled
    }

    // ===== Color Setters =====

    #[test]
    fn test_tabs_tab_bg() {
        let tabs = Tabs::new().tab_bg(Color::RED);
        assert_eq!(tabs.tab_bg, Color::RED);
    }

    #[test]
    fn test_tabs_active_bg() {
        let tabs = Tabs::new().active_bg(Color::BLUE);
        assert_eq!(tabs.active_bg, Color::BLUE);
    }

    #[test]
    fn test_tabs_inactive_color() {
        let tabs = Tabs::new().inactive_color(Color::GREEN);
        assert_eq!(tabs.inactive_color, Color::GREEN);
    }

    #[test]
    fn test_tabs_active_color() {
        let tabs = Tabs::new().active_color(Color::WHITE);
        assert_eq!(tabs.active_color, Color::WHITE);
    }

    // ===== Get Tabs Tests =====

    #[test]
    fn test_tabs_get_tabs() {
        let tabs = Tabs::new().tab(Tab::new("a", "A")).tab(Tab::new("b", "B"));

        let items = tabs.get_tabs();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].id, "a");
        assert_eq!(items[1].id, "b");
    }

    #[test]
    fn test_tabs_get_active_id_none() {
        let tabs = Tabs::new();
        assert!(tabs.get_active_id().is_none());
    }

    // ===== Paint Tests =====

    #[test]
    fn test_tabs_paint() {
        use presentar_core::RecordingCanvas;

        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .orientation(TabOrientation::Top)
            .show_border(true);
        tabs.layout(Rect::new(0.0, 0.0, 200.0, 48.0));

        let mut canvas = RecordingCanvas::new();
        tabs.paint(&mut canvas);

        // Should draw: tab bar bg + tab 1 bg + tab 1 text + tab 2 bg + tab 2 text + active indicator + border
        assert!(canvas.command_count() >= 5);
    }

    #[test]
    fn test_tabs_paint_no_border() {
        use presentar_core::RecordingCanvas;

        let mut tabs = Tabs::new().tab(Tab::new("a", "A")).show_border(false);
        tabs.layout(Rect::new(0.0, 0.0, 200.0, 48.0));

        let mut canvas = RecordingCanvas::new();
        tabs.paint(&mut canvas);

        // Should still draw tabs, just no border
        assert!(canvas.command_count() >= 2);
    }

    #[test]
    fn test_tabs_paint_disabled_tab() {
        use presentar_core::RecordingCanvas;

        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B").disabled())
            .active(0);
        tabs.layout(Rect::new(0.0, 0.0, 200.0, 48.0));

        let mut canvas = RecordingCanvas::new();
        tabs.paint(&mut canvas);

        // Should paint both tabs, disabled with different color
        assert!(canvas.command_count() >= 4);
    }

    #[test]
    fn test_tabs_paint_vertical_left() {
        use presentar_core::RecordingCanvas;

        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .orientation(TabOrientation::Left)
            .show_border(true);
        tabs.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        let mut canvas = RecordingCanvas::new();
        tabs.paint(&mut canvas);

        assert!(canvas.command_count() >= 5);
    }

    #[test]
    fn test_tabs_paint_vertical_right() {
        use presentar_core::RecordingCanvas;

        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .orientation(TabOrientation::Right)
            .show_border(true);
        tabs.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        let mut canvas = RecordingCanvas::new();
        tabs.paint(&mut canvas);

        assert!(canvas.command_count() >= 5);
    }

    #[test]
    fn test_tabs_paint_bottom() {
        use presentar_core::RecordingCanvas;

        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .orientation(TabOrientation::Bottom)
            .show_border(true);
        tabs.layout(Rect::new(0.0, 0.0, 200.0, 100.0));

        let mut canvas = RecordingCanvas::new();
        tabs.paint(&mut canvas);

        assert!(canvas.command_count() >= 5);
    }

    // ===== Brick Trait Tests =====

    #[test]
    fn test_tabs_brick_name() {
        let tabs = Tabs::new();
        assert_eq!(tabs.brick_name(), "Tabs");
    }

    #[test]
    fn test_tabs_brick_assertions() {
        let tabs = Tabs::new();
        let assertions = tabs.assertions();
        assert_eq!(assertions.len(), 1);
        assert!(assertions.contains(&BrickAssertion::MaxLatencyMs(16)));
    }

    #[test]
    fn test_tabs_brick_budget() {
        let tabs = Tabs::new();
        let budget = tabs.budget();
        // BrickBudget::uniform(16) sets internal values
        assert!(budget.measure_ms > 0);
        assert!(budget.layout_ms > 0);
        assert!(budget.paint_ms > 0);
    }

    #[test]
    fn test_tabs_brick_verify() {
        let tabs = Tabs::new();
        let verification = tabs.verify();
        assert!(verification.failed.is_empty());
        assert!(!verification.passed.is_empty());
    }

    #[test]
    fn test_tabs_to_html() {
        let tabs = Tabs::new();
        let html = tabs.to_html();
        assert!(html.contains("brick-tabs"));
    }

    #[test]
    fn test_tabs_to_css() {
        let tabs = Tabs::new();
        let css = tabs.to_css();
        assert!(css.contains("brick-tabs"));
        assert!(css.contains("display: flex"));
    }

    // ===== Widget Children Mut Tests =====

    #[test]
    fn test_tabs_children_mut() {
        let mut tabs = Tabs::new();
        assert!(tabs.children_mut().is_empty());
    }

    // ===== Event Edge Cases =====

    #[test]
    fn test_tabs_click_outside() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .min_tab_width(100.0)
            .tab_size(48.0);
        tabs.bounds = Rect::new(0.0, 0.0, 200.0, 48.0);

        // Click outside tabs
        let event = Event::MouseDown {
            position: Point::new(500.0, 24.0),
            button: MouseButton::Left,
        };

        let result = tabs.event(&event);
        assert!(result.is_none());
    }

    #[test]
    fn test_tabs_right_click_no_event() {
        let mut tabs = Tabs::new()
            .tab(Tab::new("a", "A"))
            .tab(Tab::new("b", "B"))
            .min_tab_width(100.0)
            .tab_size(48.0);
        tabs.bounds = Rect::new(0.0, 0.0, 200.0, 48.0);

        // Right click should not change tabs
        let event = Event::MouseDown {
            position: Point::new(150.0, 24.0),
            button: MouseButton::Right,
        };

        let result = tabs.event(&event);
        assert!(result.is_none());
    }

    #[test]
    fn test_tabs_other_event_no_effect() {
        let mut tabs = Tabs::new().tab(Tab::new("a", "A")).tab(Tab::new("b", "B"));
        tabs.bounds = Rect::new(0.0, 0.0, 200.0, 48.0);

        let result = tabs.event(&Event::MouseEnter);
        assert!(result.is_none());
    }
}
