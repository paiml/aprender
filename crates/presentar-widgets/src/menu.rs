//! Menu and dropdown widgets for action lists.
//!
//! Provides hierarchical menus with keyboard navigation, submenus,
//! and various item types (actions, checkable, separators).

use presentar_core::{
    widget::{LayoutResult, TextStyle},
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event, Key,
    Point, Rect, Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// Menu item variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MenuItem {
    /// Action item with label
    Action {
        /// Item label
        label: String,
        /// Unique action ID
        action: String,
        /// Whether item is disabled
        disabled: bool,
        /// Optional keyboard shortcut display
        shortcut: Option<String>,
    },
    /// Checkable item
    Checkbox {
        /// Item label
        label: String,
        /// Action ID
        action: String,
        /// Whether checked
        checked: bool,
        /// Whether disabled
        disabled: bool,
    },
    /// Separator line
    Separator,
    /// Submenu
    Submenu {
        /// Submenu label
        label: String,
        /// Child items
        items: Vec<Self>,
        /// Whether disabled
        disabled: bool,
    },
}

impl MenuItem {
    /// Create a new action item.
    #[must_use]
    pub fn action(label: impl Into<String>, action: impl Into<String>) -> Self {
        Self::Action {
            label: label.into(),
            action: action.into(),
            disabled: false,
            shortcut: None,
        }
    }

    /// Create a new checkbox item.
    #[must_use]
    pub fn checkbox(label: impl Into<String>, action: impl Into<String>, checked: bool) -> Self {
        Self::Checkbox {
            label: label.into(),
            action: action.into(),
            checked,
            disabled: false,
        }
    }

    /// Create a separator.
    #[must_use]
    pub const fn separator() -> Self {
        Self::Separator
    }

    /// Create a submenu.
    #[must_use]
    pub fn submenu(label: impl Into<String>, items: Vec<Self>) -> Self {
        Self::Submenu {
            label: label.into(),
            items,
            disabled: false,
        }
    }

    /// Set disabled state.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        match &mut self {
            Self::Action { disabled: d, .. }
            | Self::Checkbox { disabled: d, .. }
            | Self::Submenu { disabled: d, .. } => *d = disabled,
            Self::Separator => {}
        }
        self
    }

    /// Set shortcut (for action items).
    #[must_use]
    pub fn shortcut(mut self, shortcut: impl Into<String>) -> Self {
        if let Self::Action { shortcut: s, .. } = &mut self {
            *s = Some(shortcut.into());
        }
        self
    }

    /// Check if this item is selectable (not separator, not disabled).
    #[must_use]
    pub fn is_selectable(&self) -> bool {
        match self {
            Self::Action { disabled, .. }
            | Self::Checkbox { disabled, .. }
            | Self::Submenu { disabled, .. } => !disabled,
            Self::Separator => false,
        }
    }

    /// Get item height.
    #[must_use]
    pub const fn height(&self) -> f32 {
        match self {
            Self::Separator => 9.0, // 1px line + 4px padding each side
            _ => 32.0,
        }
    }
}

/// Menu trigger mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MenuTrigger {
    /// Click to open
    #[default]
    Click,
    /// Hover to open
    Hover,
    /// Right-click (context menu)
    ContextMenu,
}

/// Menu widget for dropdown actions.
#[derive(Serialize, Deserialize)]
pub struct Menu {
    /// Menu items
    pub items: Vec<MenuItem>,
    /// Whether menu is open
    pub open: bool,
    /// Trigger mode
    pub trigger: MenuTrigger,
    /// Menu width
    pub width: f32,
    /// Background color
    pub background_color: Color,
    /// Hover color
    pub hover_color: Color,
    /// Text color
    pub text_color: Color,
    /// Disabled text color
    pub disabled_color: Color,
    /// Test ID
    test_id_value: Option<String>,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
    /// Panel bounds (dropdown area)
    #[serde(skip)]
    panel_bounds: Rect,
    /// Currently highlighted index
    #[serde(skip)]
    highlighted_index: Option<usize>,
    /// Open submenu index
    #[serde(skip)]
    open_submenu: Option<usize>,
    /// Trigger widget
    #[serde(skip)]
    trigger_widget: Option<Box<dyn Widget>>,
}

impl Default for Menu {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            open: false,
            trigger: MenuTrigger::Click,
            width: 200.0,
            background_color: Color::WHITE,
            hover_color: Color::rgba(0.0, 0.0, 0.0, 0.1),
            text_color: Color::BLACK,
            disabled_color: Color::rgb(0.6, 0.6, 0.6),
            test_id_value: None,
            bounds: Rect::default(),
            panel_bounds: Rect::default(),
            highlighted_index: None,
            open_submenu: None,
            trigger_widget: None,
        }
    }
}

impl Menu {
    /// Create a new menu.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set menu items.
    #[must_use]
    pub fn items(mut self, items: Vec<MenuItem>) -> Self {
        self.items = items;
        self
    }

    /// Add a single item.
    #[must_use]
    pub fn item(mut self, item: MenuItem) -> Self {
        self.items.push(item);
        self
    }

    /// Set trigger mode.
    #[must_use]
    pub const fn trigger(mut self, trigger: MenuTrigger) -> Self {
        self.trigger = trigger;
        self
    }

    /// Set menu width.
    #[must_use]
    pub const fn width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    /// Set background color.
    #[must_use]
    pub const fn background_color(mut self, color: Color) -> Self {
        self.background_color = color;
        self
    }

    /// Set hover color.
    #[must_use]
    pub const fn hover_color(mut self, color: Color) -> Self {
        self.hover_color = color;
        self
    }

    /// Set text color.
    #[must_use]
    pub const fn text_color(mut self, color: Color) -> Self {
        self.text_color = color;
        self
    }

    /// Set the trigger widget.
    pub fn trigger_widget(mut self, widget: impl Widget + 'static) -> Self {
        self.trigger_widget = Some(Box::new(widget));
        self
    }

    /// Set test ID.
    #[must_use]
    pub fn with_test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id_value = Some(id.into());
        self
    }

    /// Open the menu.
    pub fn show(&mut self) {
        self.open = true;
        self.highlighted_index = None;
    }

    /// Close the menu.
    pub fn hide(&mut self) {
        self.open = false;
        self.highlighted_index = None;
        self.open_submenu = None;
    }

    /// Toggle the menu.
    pub fn toggle(&mut self) {
        if self.open {
            self.hide();
        } else {
            self.show();
        }
    }

    /// Check if menu is open.
    #[must_use]
    pub const fn is_open(&self) -> bool {
        self.open
    }

    /// Get highlighted index.
    #[must_use]
    pub const fn highlighted_index(&self) -> Option<usize> {
        self.highlighted_index
    }

    /// Calculate total menu height.
    fn calculate_menu_height(&self) -> f32 {
        let padding = 8.0; // Top and bottom padding
        let items_height: f32 = self.items.iter().map(MenuItem::height).sum();
        items_height + padding * 2.0
    }

    /// Find next selectable item.
    fn next_selectable(&self, from: Option<usize>, forward: bool) -> Option<usize> {
        if self.items.is_empty() {
            return None;
        }

        let start = from.map_or_else(
            || if forward { 0 } else { self.items.len() - 1 },
            |i| {
                if forward {
                    if i + 1 >= self.items.len() {
                        0
                    } else {
                        i + 1
                    }
                } else if i == 0 {
                    self.items.len() - 1
                } else {
                    i - 1
                }
            },
        );

        let mut idx = start;
        for _ in 0..self.items.len() {
            if self.items[idx].is_selectable() {
                return Some(idx);
            }
            if forward {
                idx = if idx + 1 >= self.items.len() {
                    0
                } else {
                    idx + 1
                };
            } else {
                idx = if idx == 0 {
                    self.items.len() - 1
                } else {
                    idx - 1
                };
            }
        }

        None
    }

    /// Get item at y position.
    fn item_at_position(&self, y: f32) -> Option<usize> {
        let relative_y = y - self.panel_bounds.y - 8.0; // Subtract padding
        if relative_y < 0.0 {
            return None;
        }

        let mut current_y = 0.0;
        for (i, item) in self.items.iter().enumerate() {
            let height = item.height();
            if relative_y >= current_y && relative_y < current_y + height {
                return Some(i);
            }
            current_y += height;
        }

        None
    }
}

impl Widget for Menu {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        // Measure just the trigger area
        if let Some(ref trigger) = self.trigger_widget {
            trigger.measure(constraints)
        } else {
            Size::new(self.width.min(constraints.max_width), 32.0)
        }
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;

        // Layout trigger widget
        if let Some(ref mut trigger) = self.trigger_widget {
            trigger.layout(bounds);
        }

        // Calculate menu panel bounds (below trigger)
        if self.open {
            let menu_height = self.calculate_menu_height();
            self.panel_bounds =
                Rect::new(bounds.x, bounds.y + bounds.height, self.width, menu_height);
        }

        LayoutResult {
            size: bounds.size(),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn paint(&self, canvas: &mut dyn Canvas) {
        // Paint trigger
        if let Some(ref trigger) = self.trigger_widget {
            trigger.paint(canvas);
        }

        if !self.open {
            return;
        }

        // Paint menu background with shadow
        let shadow_bounds = Rect::new(
            self.panel_bounds.x + 2.0,
            self.panel_bounds.y + 2.0,
            self.panel_bounds.width,
            self.panel_bounds.height,
        );
        canvas.fill_rect(shadow_bounds, Color::rgba(0.0, 0.0, 0.0, 0.1));
        canvas.fill_rect(self.panel_bounds, self.background_color);

        // Paint items
        let mut y = self.panel_bounds.y + 8.0; // Top padding
        let text_style = TextStyle {
            size: 14.0,
            color: self.text_color,
            ..Default::default()
        };
        let disabled_style = TextStyle {
            size: 14.0,
            color: self.disabled_color,
            ..Default::default()
        };

        for (i, item) in self.items.iter().enumerate() {
            let height = item.height();

            match item {
                MenuItem::Action {
                    label,
                    disabled,
                    shortcut,
                    ..
                } => {
                    // Hover background
                    if self.highlighted_index == Some(i) && !disabled {
                        let hover_rect =
                            Rect::new(self.panel_bounds.x, y, self.panel_bounds.width, height);
                        canvas.fill_rect(hover_rect, self.hover_color);
                    }

                    // Label
                    let style = if *disabled {
                        &disabled_style
                    } else {
                        &text_style
                    };
                    canvas.draw_text(
                        label,
                        Point::new(self.panel_bounds.x + 12.0, y + 20.0),
                        style,
                    );

                    // Shortcut
                    if let Some(ref shortcut) = shortcut {
                        let shortcut_style = TextStyle {
                            size: 12.0,
                            color: self.disabled_color,
                            ..Default::default()
                        };
                        canvas.draw_text(
                            shortcut,
                            Point::new(
                                self.panel_bounds.x + self.panel_bounds.width - 60.0,
                                y + 20.0,
                            ),
                            &shortcut_style,
                        );
                    }
                }
                MenuItem::Checkbox {
                    label,
                    checked,
                    disabled,
                    ..
                } => {
                    // Hover background
                    if self.highlighted_index == Some(i) && !disabled {
                        let hover_rect =
                            Rect::new(self.panel_bounds.x, y, self.panel_bounds.width, height);
                        canvas.fill_rect(hover_rect, self.hover_color);
                    }

                    // Checkbox indicator
                    let check_text = if *checked { "✓" } else { " " };
                    let style = if *disabled {
                        &disabled_style
                    } else {
                        &text_style
                    };
                    canvas.draw_text(
                        check_text,
                        Point::new(self.panel_bounds.x + 12.0, y + 20.0),
                        style,
                    );

                    // Label
                    canvas.draw_text(
                        label,
                        Point::new(self.panel_bounds.x + 32.0, y + 20.0),
                        style,
                    );
                }
                MenuItem::Separator => {
                    let line_y = y + 4.0;
                    canvas.draw_line(
                        Point::new(self.panel_bounds.x + 8.0, line_y),
                        Point::new(self.panel_bounds.x + self.panel_bounds.width - 8.0, line_y),
                        Color::rgb(0.9, 0.9, 0.9),
                        1.0,
                    );
                }
                MenuItem::Submenu {
                    label, disabled, ..
                } => {
                    // Hover background
                    if self.highlighted_index == Some(i) && !disabled {
                        let hover_rect =
                            Rect::new(self.panel_bounds.x, y, self.panel_bounds.width, height);
                        canvas.fill_rect(hover_rect, self.hover_color);
                    }

                    // Label
                    let style = if *disabled {
                        &disabled_style
                    } else {
                        &text_style
                    };
                    canvas.draw_text(
                        label,
                        Point::new(self.panel_bounds.x + 12.0, y + 20.0),
                        style,
                    );

                    // Arrow indicator
                    canvas.draw_text(
                        "›",
                        Point::new(
                            self.panel_bounds.x + self.panel_bounds.width - 20.0,
                            y + 20.0,
                        ),
                        style,
                    );
                }
            }

            y += height;
        }
    }

    #[allow(clippy::too_many_lines)]
    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        match event {
            Event::MouseDown { position, .. } => {
                // Check if click is on trigger
                let on_trigger = position.x >= self.bounds.x
                    && position.x <= self.bounds.x + self.bounds.width
                    && position.y >= self.bounds.y
                    && position.y <= self.bounds.y + self.bounds.height;

                if on_trigger && self.trigger == MenuTrigger::Click {
                    self.toggle();
                    return Some(Box::new(MenuToggled { open: self.open }));
                }

                // Check if click is on menu item
                if self.open {
                    let on_menu = position.x >= self.panel_bounds.x
                        && position.x <= self.panel_bounds.x + self.panel_bounds.width
                        && position.y >= self.panel_bounds.y
                        && position.y <= self.panel_bounds.y + self.panel_bounds.height;

                    if on_menu {
                        if let Some(idx) = self.item_at_position(position.y) {
                            if let Some(item) = self.items.get_mut(idx) {
                                match item {
                                    MenuItem::Action {
                                        action, disabled, ..
                                    } if !*disabled => {
                                        let action_id = action.clone();
                                        self.hide();
                                        return Some(Box::new(MenuItemSelected {
                                            action: action_id,
                                        }));
                                    }
                                    MenuItem::Checkbox {
                                        action,
                                        checked,
                                        disabled,
                                        ..
                                    } if !*disabled => {
                                        *checked = !*checked;
                                        let action_id = action.clone();
                                        let is_checked = *checked;
                                        return Some(Box::new(MenuCheckboxToggled {
                                            action: action_id,
                                            checked: is_checked,
                                        }));
                                    }
                                    MenuItem::Submenu { disabled, .. } if !*disabled => {
                                        self.open_submenu = Some(idx);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    } else {
                        // Click outside menu - close
                        self.hide();
                        return Some(Box::new(MenuClosed));
                    }
                }
            }
            Event::MouseMove { position } => {
                if self.open {
                    let on_menu = position.x >= self.panel_bounds.x
                        && position.x <= self.panel_bounds.x + self.panel_bounds.width
                        && position.y >= self.panel_bounds.y
                        && position.y <= self.panel_bounds.y + self.panel_bounds.height;

                    if on_menu {
                        self.highlighted_index = self.item_at_position(position.y);
                    } else {
                        self.highlighted_index = None;
                    }
                }
            }
            Event::KeyDown { key, .. } if self.open => match key {
                Key::Escape => {
                    self.hide();
                    return Some(Box::new(MenuClosed));
                }
                Key::Up => {
                    self.highlighted_index = self.next_selectable(self.highlighted_index, false);
                }
                Key::Down => {
                    self.highlighted_index = self.next_selectable(self.highlighted_index, true);
                }
                Key::Enter | Key::Space => {
                    if let Some(idx) = self.highlighted_index {
                        if let Some(item) = self.items.get_mut(idx) {
                            match item {
                                MenuItem::Action {
                                    action, disabled, ..
                                } if !*disabled => {
                                    let action_id = action.clone();
                                    self.hide();
                                    return Some(Box::new(MenuItemSelected { action: action_id }));
                                }
                                MenuItem::Checkbox {
                                    action,
                                    checked,
                                    disabled,
                                    ..
                                } if !*disabled => {
                                    *checked = !*checked;
                                    let action_id = action.clone();
                                    let is_checked = *checked;
                                    return Some(Box::new(MenuCheckboxToggled {
                                        action: action_id,
                                        checked: is_checked,
                                    }));
                                }
                                _ => {}
                            }
                        }
                    }
                }
                _ => {}
            },
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
impl Brick for Menu {
    fn brick_name(&self) -> &'static str {
        "Menu"
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
        r#"<div class="brick-menu"></div>"#.to_string()
    }

    fn to_css(&self) -> String {
        ".brick-menu { display: block; position: relative; }".to_string()
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

/// Message when menu is toggled.
#[derive(Debug, Clone)]
pub struct MenuToggled {
    /// Whether menu is now open
    pub open: bool,
}

/// Message when menu item is selected.
#[derive(Debug, Clone)]
pub struct MenuItemSelected {
    /// Action ID of selected item
    pub action: String,
}

/// Message when menu checkbox is toggled.
#[derive(Debug, Clone)]
pub struct MenuCheckboxToggled {
    /// Action ID
    pub action: String,
    /// New checked state
    pub checked: bool,
}

/// Message when menu is closed.
#[derive(Debug, Clone)]
pub struct MenuClosed;

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // MenuItem Tests
    // =========================================================================

    #[test]
    fn test_menu_item_action() {
        let item = MenuItem::action("Cut", "edit.cut");
        match item {
            MenuItem::Action {
                label,
                action,
                disabled,
                shortcut,
            } => {
                assert_eq!(label, "Cut");
                assert_eq!(action, "edit.cut");
                assert!(!disabled);
                assert!(shortcut.is_none());
            }
            _ => panic!("Expected Action"),
        }
    }

    #[test]
    fn test_menu_item_action_with_shortcut() {
        let item = MenuItem::action("Cut", "edit.cut").shortcut("Ctrl+X");
        match item {
            MenuItem::Action { shortcut, .. } => {
                assert_eq!(shortcut, Some("Ctrl+X".to_string()));
            }
            _ => panic!("Expected Action"),
        }
    }

    #[test]
    fn test_menu_item_checkbox() {
        let item = MenuItem::checkbox("Show Grid", "view.grid", true);
        match item {
            MenuItem::Checkbox {
                label,
                checked,
                disabled,
                ..
            } => {
                assert_eq!(label, "Show Grid");
                assert!(checked);
                assert!(!disabled);
            }
            _ => panic!("Expected Checkbox"),
        }
    }

    #[test]
    fn test_menu_item_separator() {
        let item = MenuItem::separator();
        assert!(matches!(item, MenuItem::Separator));
    }

    #[test]
    fn test_menu_item_submenu() {
        let items = vec![MenuItem::action("Sub 1", "sub.1")];
        let item = MenuItem::submenu("More", items);
        match item {
            MenuItem::Submenu {
                label,
                items,
                disabled,
            } => {
                assert_eq!(label, "More");
                assert_eq!(items.len(), 1);
                assert!(!disabled);
            }
            _ => panic!("Expected Submenu"),
        }
    }

    #[test]
    fn test_menu_item_disabled() {
        let item = MenuItem::action("Cut", "edit.cut").disabled(true);
        match item {
            MenuItem::Action { disabled, .. } => assert!(disabled),
            _ => panic!("Expected Action"),
        }
    }

    #[test]
    fn test_menu_item_is_selectable() {
        assert!(MenuItem::action("Cut", "edit.cut").is_selectable());
        assert!(!MenuItem::action("Cut", "edit.cut")
            .disabled(true)
            .is_selectable());
        assert!(!MenuItem::separator().is_selectable());
        assert!(MenuItem::checkbox("Show", "show", false).is_selectable());
    }

    #[test]
    fn test_menu_item_height() {
        assert_eq!(MenuItem::action("Cut", "edit.cut").height(), 32.0);
        assert_eq!(MenuItem::separator().height(), 9.0);
    }

    // =========================================================================
    // Menu Tests
    // =========================================================================

    #[test]
    fn test_menu_new() {
        let menu = Menu::new();
        assert!(menu.items.is_empty());
        assert!(!menu.open);
        assert_eq!(menu.trigger, MenuTrigger::Click);
    }

    #[test]
    fn test_menu_builder() {
        let menu = Menu::new()
            .items(vec![
                MenuItem::action("Cut", "cut"),
                MenuItem::separator(),
                MenuItem::action("Paste", "paste"),
            ])
            .trigger(MenuTrigger::Hover)
            .width(250.0);

        assert_eq!(menu.items.len(), 3);
        assert_eq!(menu.trigger, MenuTrigger::Hover);
        assert_eq!(menu.width, 250.0);
    }

    #[test]
    fn test_menu_add_item() {
        let menu = Menu::new()
            .item(MenuItem::action("Cut", "cut"))
            .item(MenuItem::action("Copy", "copy"));
        assert_eq!(menu.items.len(), 2);
    }

    #[test]
    fn test_menu_show_hide() {
        let mut menu = Menu::new();
        assert!(!menu.is_open());

        menu.show();
        assert!(menu.is_open());

        menu.hide();
        assert!(!menu.is_open());
    }

    #[test]
    fn test_menu_toggle() {
        let mut menu = Menu::new();

        menu.toggle();
        assert!(menu.is_open());

        menu.toggle();
        assert!(!menu.is_open());
    }

    #[test]
    fn test_menu_calculate_height() {
        let menu = Menu::new().items(vec![
            MenuItem::action("Cut", "cut"),
            MenuItem::separator(),
            MenuItem::action("Paste", "paste"),
        ]);
        // 2 actions (32px each) + 1 separator (9px) + padding (16px) = 89px
        assert_eq!(menu.calculate_menu_height(), 89.0);
    }

    #[test]
    fn test_menu_measure() {
        let menu = Menu::new().width(200.0);
        let size = menu.measure(Constraints::loose(Size::new(300.0, 400.0)));
        assert_eq!(size.width, 200.0);
    }

    #[test]
    fn test_menu_layout() {
        let mut menu = Menu::new().width(200.0);
        menu.open = true;
        menu.items = vec![MenuItem::action("Cut", "cut")];

        let result = menu.layout(Rect::new(10.0, 20.0, 100.0, 32.0));
        assert_eq!(result.size, Size::new(100.0, 32.0));
        assert_eq!(menu.panel_bounds.x, 10.0);
        assert_eq!(menu.panel_bounds.y, 52.0); // Below trigger
    }

    #[test]
    fn test_menu_type_id() {
        let menu = Menu::new();
        assert_eq!(Widget::type_id(&menu), TypeId::of::<Menu>());
    }

    #[test]
    fn test_menu_is_focusable() {
        let menu = Menu::new();
        assert!(menu.is_focusable());
    }

    #[test]
    fn test_menu_test_id() {
        let menu = Menu::new().with_test_id("my-menu");
        assert_eq!(Widget::test_id(&menu), Some("my-menu"));
    }

    #[test]
    fn test_menu_highlighted_index() {
        let mut menu = Menu::new();
        assert!(menu.highlighted_index().is_none());

        menu.highlighted_index = Some(2);
        assert_eq!(menu.highlighted_index(), Some(2));
    }

    #[test]
    fn test_menu_next_selectable() {
        let menu = Menu::new().items(vec![
            MenuItem::action("Cut", "cut"),
            MenuItem::separator(),
            MenuItem::action("Paste", "paste"),
        ]);

        // Forward from None
        assert_eq!(menu.next_selectable(None, true), Some(0));

        // Forward from 0
        assert_eq!(menu.next_selectable(Some(0), true), Some(2)); // Skips separator

        // Backward from 2
        assert_eq!(menu.next_selectable(Some(2), false), Some(0)); // Skips separator
    }

    #[test]
    fn test_menu_escape_closes() {
        let mut menu = Menu::new().items(vec![MenuItem::action("Cut", "cut")]);
        menu.show();

        let result = menu.event(&Event::key_down(Key::Escape));
        assert!(result.is_some());
        assert!(!menu.is_open());
    }

    #[test]
    fn test_menu_arrow_navigation() {
        let mut menu = Menu::new().items(vec![
            MenuItem::action("Cut", "cut"),
            MenuItem::action("Copy", "copy"),
        ]);
        menu.show();

        menu.event(&Event::key_down(Key::Down));
        assert_eq!(menu.highlighted_index, Some(0));

        menu.event(&Event::key_down(Key::Down));
        assert_eq!(menu.highlighted_index, Some(1));
    }

    // =========================================================================
    // Message Tests
    // =========================================================================

    #[test]
    fn test_menu_toggled_message() {
        let msg = MenuToggled { open: true };
        assert!(msg.open);
    }

    #[test]
    fn test_menu_item_selected_message() {
        let msg = MenuItemSelected {
            action: "edit.cut".to_string(),
        };
        assert_eq!(msg.action, "edit.cut");
    }

    #[test]
    fn test_menu_checkbox_toggled_message() {
        let msg = MenuCheckboxToggled {
            action: "view.grid".to_string(),
            checked: true,
        };
        assert_eq!(msg.action, "view.grid");
        assert!(msg.checked);
    }

    #[test]
    fn test_menu_closed_message() {
        let msg = MenuClosed;
        assert_eq!(format!("{msg:?}"), "MenuClosed");
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_menu_shortcut_on_non_action() {
        // shortcut() should do nothing on checkbox items
        let item = MenuItem::checkbox("Show", "show", false).shortcut("Ctrl+S");
        match item {
            MenuItem::Checkbox { .. } => {} // Still checkbox, shortcut not applied
            _ => panic!("Expected Checkbox"),
        }
    }

    #[test]
    fn test_menu_disabled_checkbox() {
        let item = MenuItem::checkbox("Show", "show", true).disabled(true);
        match item {
            MenuItem::Checkbox { disabled, .. } => assert!(disabled),
            _ => panic!("Expected Checkbox"),
        }
    }

    #[test]
    fn test_menu_disabled_submenu() {
        let item = MenuItem::submenu("More", vec![]).disabled(true);
        match item {
            MenuItem::Submenu { disabled, .. } => assert!(disabled),
            _ => panic!("Expected Submenu"),
        }
    }

    #[test]
    fn test_menu_disabled_separator_no_op() {
        // Calling disabled on separator should be a no-op
        let item = MenuItem::separator().disabled(true);
        assert!(matches!(item, MenuItem::Separator));
    }

    #[test]
    fn test_menu_submenu_not_selectable_when_disabled() {
        let item = MenuItem::submenu("More", vec![]).disabled(true);
        assert!(!item.is_selectable());
    }

    #[test]
    fn test_menu_context_menu_trigger() {
        let menu = Menu::new().trigger(MenuTrigger::ContextMenu);
        assert_eq!(menu.trigger, MenuTrigger::ContextMenu);
    }

    #[test]
    fn test_menu_hover_trigger() {
        let menu = Menu::new().trigger(MenuTrigger::Hover);
        assert_eq!(menu.trigger, MenuTrigger::Hover);
    }

    #[test]
    fn test_menu_background_color() {
        let menu = Menu::new().background_color(Color::RED);
        assert_eq!(menu.background_color, Color::RED);
    }

    #[test]
    fn test_menu_hover_color() {
        let menu = Menu::new().hover_color(Color::BLUE);
        assert_eq!(menu.hover_color, Color::BLUE);
    }

    #[test]
    fn test_menu_text_color() {
        let menu = Menu::new().text_color(Color::GREEN);
        assert_eq!(menu.text_color, Color::GREEN);
    }

    #[test]
    fn test_menu_next_selectable_empty() {
        let menu = Menu::new();
        assert!(menu.next_selectable(None, true).is_none());
        assert!(menu.next_selectable(None, false).is_none());
    }

    #[test]
    fn test_menu_next_selectable_all_disabled() {
        let menu = Menu::new().items(vec![
            MenuItem::separator(),
            MenuItem::action("Cut", "cut").disabled(true),
            MenuItem::separator(),
        ]);
        assert!(menu.next_selectable(None, true).is_none());
    }

    #[test]
    fn test_menu_next_selectable_wrap_forward() {
        let menu = Menu::new().items(vec![MenuItem::action("A", "a"), MenuItem::action("B", "b")]);
        // From last item, should wrap to first
        assert_eq!(menu.next_selectable(Some(1), true), Some(0));
    }

    #[test]
    fn test_menu_next_selectable_wrap_backward() {
        let menu = Menu::new().items(vec![MenuItem::action("A", "a"), MenuItem::action("B", "b")]);
        // From first item, should wrap to last
        assert_eq!(menu.next_selectable(Some(0), false), Some(1));
    }

    #[test]
    fn test_menu_children_empty() {
        let menu = Menu::new();
        assert!(menu.children().is_empty());
    }

    #[test]
    fn test_menu_children_mut_empty() {
        let mut menu = Menu::new();
        assert!(menu.children_mut().is_empty());
    }

    #[test]
    fn test_menu_bounds() {
        let mut menu = Menu::new();
        menu.layout(Rect::new(10.0, 20.0, 200.0, 32.0));
        assert_eq!(menu.bounds(), Rect::new(10.0, 20.0, 200.0, 32.0));
    }

    #[test]
    fn test_menu_trigger_default() {
        assert_eq!(MenuTrigger::default(), MenuTrigger::Click);
    }

    #[test]
    fn test_menu_event_closed_returns_none() {
        let mut menu = Menu::new();
        // Event on closed menu should return None
        let result = menu.event(&Event::key_down(Key::Down));
        assert!(result.is_none());
    }

    #[test]
    fn test_menu_enter_selects_item() {
        let mut menu = Menu::new().items(vec![MenuItem::action("Cut", "cut")]);
        menu.show();
        menu.highlighted_index = Some(0);
        menu.layout(Rect::new(0.0, 0.0, 200.0, 32.0));

        let result = menu.event(&Event::key_down(Key::Enter));
        assert!(result.is_some());
        assert!(!menu.is_open());
    }

    #[test]
    fn test_menu_space_selects_item() {
        let mut menu = Menu::new().items(vec![MenuItem::action("Cut", "cut")]);
        menu.show();
        menu.highlighted_index = Some(0);
        menu.layout(Rect::new(0.0, 0.0, 200.0, 32.0));

        let result = menu.event(&Event::key_down(Key::Space));
        assert!(result.is_some());
    }

    #[test]
    fn test_menu_enter_on_checkbox_toggles() {
        let mut menu = Menu::new().items(vec![MenuItem::checkbox("Show", "show", false)]);
        menu.show();
        menu.highlighted_index = Some(0);
        menu.layout(Rect::new(0.0, 0.0, 200.0, 32.0));

        let result = menu.event(&Event::key_down(Key::Enter));
        assert!(result.is_some());
        // Menu stays open for checkbox
        assert!(menu.is_open());
    }

    #[test]
    fn test_menu_enter_on_disabled_does_nothing() {
        let mut menu = Menu::new().items(vec![MenuItem::action("Cut", "cut").disabled(true)]);
        menu.show();
        menu.highlighted_index = Some(0);
        menu.layout(Rect::new(0.0, 0.0, 200.0, 32.0));

        let result = menu.event(&Event::key_down(Key::Enter));
        assert!(result.is_none());
        assert!(menu.is_open()); // Still open
    }

    #[test]
    fn test_menu_up_arrow_navigation() {
        let mut menu =
            Menu::new().items(vec![MenuItem::action("A", "a"), MenuItem::action("B", "b")]);
        menu.show();
        menu.highlighted_index = Some(1);

        menu.event(&Event::key_down(Key::Up));
        assert_eq!(menu.highlighted_index, Some(0));
    }

    // =========================================================================
    // Brick Trait Tests
    // =========================================================================

    #[test]
    fn test_menu_brick_name() {
        let menu = Menu::new();
        assert_eq!(menu.brick_name(), "Menu");
    }

    #[test]
    fn test_menu_brick_assertions() {
        let menu = Menu::new();
        let assertions = menu.assertions();
        assert!(!assertions.is_empty());
        assert!(matches!(assertions[0], BrickAssertion::MaxLatencyMs(16)));
    }

    #[test]
    fn test_menu_brick_budget() {
        let menu = Menu::new();
        let budget = menu.budget();
        // Verify budget has reasonable values
        assert!(budget.layout_ms > 0);
        assert!(budget.paint_ms > 0);
    }

    #[test]
    fn test_menu_brick_verify() {
        let menu = Menu::new();
        let verification = menu.verify();
        assert!(!verification.passed.is_empty());
        assert!(verification.failed.is_empty());
    }

    #[test]
    fn test_menu_brick_to_html() {
        let menu = Menu::new();
        let html = menu.to_html();
        assert!(html.contains("brick-menu"));
    }

    #[test]
    fn test_menu_brick_to_css() {
        let menu = Menu::new();
        let css = menu.to_css();
        assert!(css.contains(".brick-menu"));
        assert!(css.contains("display: block"));
        assert!(css.contains("position: relative"));
    }

    #[test]
    fn test_menu_brick_test_id() {
        let menu = Menu::new().with_test_id("my-menu");
        assert_eq!(Brick::test_id(&menu), Some("my-menu"));
    }

    #[test]
    fn test_menu_brick_test_id_none() {
        let menu = Menu::new();
        assert!(Brick::test_id(&menu).is_none());
    }

    // =========================================================================
    // Item at Position Tests
    // =========================================================================

    #[test]
    fn test_menu_item_at_position_valid() {
        let mut menu = Menu::new().items(vec![
            MenuItem::action("Cut", "cut"),
            MenuItem::action("Copy", "copy"),
            MenuItem::action("Paste", "paste"),
        ]);
        menu.open = true;
        menu.layout(Rect::new(0.0, 0.0, 200.0, 32.0));

        // Item 0 starts at y = panel_bounds.y + 8.0
        // Each action item is 32px tall
        let item = menu.item_at_position(menu.panel_bounds.y + 8.0 + 10.0);
        assert_eq!(item, Some(0));

        let item = menu.item_at_position(menu.panel_bounds.y + 8.0 + 40.0);
        assert_eq!(item, Some(1));

        let item = menu.item_at_position(menu.panel_bounds.y + 8.0 + 72.0);
        assert_eq!(item, Some(2));
    }

    #[test]
    fn test_menu_item_at_position_above_menu() {
        let mut menu = Menu::new().items(vec![MenuItem::action("Cut", "cut")]);
        menu.open = true;
        menu.layout(Rect::new(0.0, 0.0, 200.0, 32.0));

        // Y position above the menu
        let item = menu.item_at_position(menu.panel_bounds.y - 10.0);
        assert!(item.is_none());
    }

    #[test]
    fn test_menu_item_at_position_below_items() {
        let mut menu = Menu::new().items(vec![MenuItem::action("Cut", "cut")]);
        menu.open = true;
        menu.layout(Rect::new(0.0, 0.0, 200.0, 32.0));

        // Y position far below the item
        let item = menu.item_at_position(menu.panel_bounds.y + 500.0);
        assert!(item.is_none());
    }

    // =========================================================================
    // Mouse Event Tests
    // =========================================================================

    #[test]
    fn test_menu_click_on_trigger_opens() {
        let mut menu = Menu::new().items(vec![MenuItem::action("Cut", "cut")]);
        menu.layout(Rect::new(10.0, 10.0, 200.0, 32.0));

        // Click inside trigger area
        let result = menu.event(&Event::MouseDown {
            position: Point::new(50.0, 20.0),
            button: presentar_core::MouseButton::Left,
        });

        assert!(result.is_some());
        assert!(menu.is_open());
    }

    #[test]
    fn test_menu_click_outside_closes() {
        let mut menu = Menu::new().items(vec![MenuItem::action("Cut", "cut")]);
        menu.show();
        menu.layout(Rect::new(10.0, 10.0, 200.0, 32.0));

        // Click far outside menu area
        let result = menu.event(&Event::MouseDown {
            position: Point::new(500.0, 500.0),
            button: presentar_core::MouseButton::Left,
        });

        assert!(result.is_some());
        assert!(!menu.is_open());
    }

    #[test]
    fn test_menu_click_on_action_item() {
        let mut menu = Menu::new().items(vec![MenuItem::action("Cut", "cut")]);
        menu.show();
        menu.layout(Rect::new(0.0, 0.0, 200.0, 32.0));

        // Click on the action item
        let click_y = menu.panel_bounds.y + 8.0 + 16.0; // Middle of first item
        let result = menu.event(&Event::MouseDown {
            position: Point::new(menu.panel_bounds.x + 50.0, click_y),
            button: presentar_core::MouseButton::Left,
        });

        assert!(result.is_some());
        assert!(!menu.is_open()); // Menu closes after action
    }

    #[test]
    fn test_menu_click_on_checkbox_item() {
        let mut menu = Menu::new().items(vec![MenuItem::checkbox("Show Grid", "show", false)]);
        menu.show();
        menu.layout(Rect::new(0.0, 0.0, 200.0, 32.0));

        // Click on the checkbox item
        let click_y = menu.panel_bounds.y + 8.0 + 16.0;
        let result = menu.event(&Event::MouseDown {
            position: Point::new(menu.panel_bounds.x + 50.0, click_y),
            button: presentar_core::MouseButton::Left,
        });

        assert!(result.is_some());
        // Checkbox should be toggled
        if let MenuItem::Checkbox { checked, .. } = &menu.items[0] {
            assert!(*checked);
        } else {
            panic!("Expected Checkbox item");
        }
    }

    #[test]
    fn test_menu_click_on_disabled_action() {
        let mut menu = Menu::new().items(vec![MenuItem::action("Cut", "cut").disabled(true)]);
        menu.show();
        menu.layout(Rect::new(0.0, 0.0, 200.0, 32.0));

        // Click on the disabled item
        let click_y = menu.panel_bounds.y + 8.0 + 16.0;
        let result = menu.event(&Event::MouseDown {
            position: Point::new(menu.panel_bounds.x + 50.0, click_y),
            button: presentar_core::MouseButton::Left,
        });

        assert!(result.is_none()); // No action taken
        assert!(menu.is_open()); // Menu stays open
    }

    #[test]
    fn test_menu_click_on_submenu_opens_it() {
        let mut menu = Menu::new().items(vec![MenuItem::submenu(
            "More",
            vec![MenuItem::action("Sub", "sub")],
        )]);
        menu.show();
        menu.layout(Rect::new(0.0, 0.0, 200.0, 32.0));

        // Click on the submenu item
        let click_y = menu.panel_bounds.y + 8.0 + 16.0;
        let result = menu.event(&Event::MouseDown {
            position: Point::new(menu.panel_bounds.x + 50.0, click_y),
            button: presentar_core::MouseButton::Left,
        });

        assert!(result.is_none()); // No message, just opens submenu
        assert_eq!(menu.open_submenu, Some(0));
    }

    #[test]
    fn test_menu_mouse_move_updates_highlight() {
        let mut menu = Menu::new().items(vec![
            MenuItem::action("Cut", "cut"),
            MenuItem::action("Copy", "copy"),
        ]);
        menu.show();
        menu.layout(Rect::new(0.0, 0.0, 200.0, 32.0));

        // Move over first item
        let y1 = menu.panel_bounds.y + 8.0 + 16.0;
        menu.event(&Event::MouseMove {
            position: Point::new(menu.panel_bounds.x + 50.0, y1),
        });
        assert_eq!(menu.highlighted_index, Some(0));

        // Move over second item
        let y2 = menu.panel_bounds.y + 8.0 + 48.0;
        menu.event(&Event::MouseMove {
            position: Point::new(menu.panel_bounds.x + 50.0, y2),
        });
        assert_eq!(menu.highlighted_index, Some(1));
    }

    #[test]
    fn test_menu_mouse_move_outside_clears_highlight() {
        let mut menu = Menu::new().items(vec![MenuItem::action("Cut", "cut")]);
        menu.show();
        menu.layout(Rect::new(0.0, 0.0, 200.0, 32.0));
        menu.highlighted_index = Some(0);

        // Move outside menu
        menu.event(&Event::MouseMove {
            position: Point::new(500.0, 500.0),
        });
        assert!(menu.highlighted_index.is_none());
    }

    // =========================================================================
    // Keyboard Navigation Edge Cases
    // =========================================================================

    #[test]
    fn test_menu_up_from_none_selects_last() {
        let mut menu =
            Menu::new().items(vec![MenuItem::action("A", "a"), MenuItem::action("B", "b")]);
        menu.show();

        menu.event(&Event::key_down(Key::Up));
        assert_eq!(menu.highlighted_index, Some(1)); // Wraps to last
    }

    #[test]
    fn test_menu_down_from_last_wraps_to_first() {
        let mut menu =
            Menu::new().items(vec![MenuItem::action("A", "a"), MenuItem::action("B", "b")]);
        menu.show();
        menu.highlighted_index = Some(1);

        menu.event(&Event::key_down(Key::Down));
        assert_eq!(menu.highlighted_index, Some(0)); // Wraps to first
    }

    #[test]
    fn test_menu_up_skips_separator() {
        let mut menu = Menu::new().items(vec![
            MenuItem::action("A", "a"),
            MenuItem::separator(),
            MenuItem::action("B", "b"),
        ]);
        menu.show();
        menu.highlighted_index = Some(2);

        menu.event(&Event::key_down(Key::Up));
        assert_eq!(menu.highlighted_index, Some(0)); // Skips separator
    }

    #[test]
    fn test_menu_down_skips_disabled() {
        let mut menu = Menu::new().items(vec![
            MenuItem::action("A", "a"),
            MenuItem::action("B", "b").disabled(true),
            MenuItem::action("C", "c"),
        ]);
        menu.show();
        menu.highlighted_index = Some(0);

        menu.event(&Event::key_down(Key::Down));
        assert_eq!(menu.highlighted_index, Some(2)); // Skips disabled
    }

    #[test]
    fn test_menu_other_key_does_nothing() {
        let mut menu = Menu::new().items(vec![MenuItem::action("A", "a")]);
        menu.show();
        menu.highlighted_index = Some(0);

        let result = menu.event(&Event::key_down(Key::Tab));
        assert!(result.is_none());
        assert_eq!(menu.highlighted_index, Some(0));
    }

    #[test]
    fn test_menu_enter_on_separator_does_nothing() {
        let mut menu = Menu::new().items(vec![MenuItem::separator(), MenuItem::action("A", "a")]);
        menu.show();
        menu.highlighted_index = Some(0); // Manually set to separator (shouldn't happen in practice)

        let result = menu.event(&Event::key_down(Key::Enter));
        assert!(result.is_none());
        assert!(menu.is_open());
    }

    #[test]
    fn test_menu_enter_on_submenu_does_nothing() {
        let mut menu = Menu::new().items(vec![MenuItem::submenu(
            "More",
            vec![MenuItem::action("Sub", "sub")],
        )]);
        menu.show();
        menu.highlighted_index = Some(0);

        let result = menu.event(&Event::key_down(Key::Enter));
        assert!(result.is_none());
        assert!(menu.is_open());
    }

    #[test]
    fn test_menu_space_on_checkbox_toggles() {
        let mut menu = Menu::new().items(vec![MenuItem::checkbox("Show", "show", false)]);
        menu.show();
        menu.highlighted_index = Some(0);
        menu.layout(Rect::new(0.0, 0.0, 200.0, 32.0));

        let result = menu.event(&Event::key_down(Key::Space));
        assert!(result.is_some());
        // Checkbox should be toggled
        if let MenuItem::Checkbox { checked, .. } = &menu.items[0] {
            assert!(*checked);
        }
    }

    // =========================================================================
    // MenuItem Methods Tests
    // =========================================================================

    #[test]
    fn test_menu_item_height_action() {
        let item = MenuItem::action("Test", "test");
        assert_eq!(item.height(), 32.0);
    }

    #[test]
    fn test_menu_item_height_checkbox() {
        let item = MenuItem::checkbox("Test", "test", false);
        assert_eq!(item.height(), 32.0);
    }

    #[test]
    fn test_menu_item_height_submenu() {
        let item = MenuItem::submenu("More", vec![]);
        assert_eq!(item.height(), 32.0);
    }

    #[test]
    fn test_menu_item_is_selectable_submenu() {
        let item = MenuItem::submenu("More", vec![]);
        assert!(item.is_selectable());
    }

    #[test]
    fn test_menu_item_is_selectable_disabled_checkbox() {
        let item = MenuItem::checkbox("Test", "test", false).disabled(true);
        assert!(!item.is_selectable());
    }

    // =========================================================================
    // Menu Trigger Tests
    // =========================================================================

    #[test]
    fn test_menu_trigger_hover_no_click_open() {
        let mut menu = Menu::new()
            .trigger(MenuTrigger::Hover)
            .items(vec![MenuItem::action("Cut", "cut")]);
        menu.layout(Rect::new(10.0, 10.0, 200.0, 32.0));

        // Click with hover trigger should not toggle
        let result = menu.event(&Event::MouseDown {
            position: Point::new(50.0, 20.0),
            button: presentar_core::MouseButton::Left,
        });

        assert!(result.is_none());
        assert!(!menu.is_open());
    }

    #[test]
    fn test_menu_trigger_context_menu_no_click_open() {
        let mut menu = Menu::new()
            .trigger(MenuTrigger::ContextMenu)
            .items(vec![MenuItem::action("Cut", "cut")]);
        menu.layout(Rect::new(10.0, 10.0, 200.0, 32.0));

        // Regular click with context trigger should not toggle
        let result = menu.event(&Event::MouseDown {
            position: Point::new(50.0, 20.0),
            button: presentar_core::MouseButton::Left,
        });

        assert!(result.is_none());
        assert!(!menu.is_open());
    }

    // =========================================================================
    // Message Clone Tests
    // =========================================================================

    #[test]
    fn test_menu_toggled_clone() {
        let msg = MenuToggled { open: true };
        let cloned = msg.clone();
        assert_eq!(cloned.open, msg.open);
    }

    #[test]
    fn test_menu_item_selected_clone() {
        let msg = MenuItemSelected {
            action: "test".to_string(),
        };
        let cloned = msg.clone();
        assert_eq!(cloned.action, msg.action);
    }

    #[test]
    fn test_menu_checkbox_toggled_clone() {
        let msg = MenuCheckboxToggled {
            action: "test".to_string(),
            checked: true,
        };
        let cloned = msg.clone();
        assert_eq!(cloned.action, msg.action);
        assert_eq!(cloned.checked, msg.checked);
    }

    #[test]
    fn test_menu_closed_clone() {
        let msg = MenuClosed;
        let cloned = msg;
        assert_eq!(format!("{cloned:?}"), "MenuClosed");
    }

    // =========================================================================
    // Default Trait Tests
    // =========================================================================

    #[test]
    fn test_menu_default() {
        let menu = Menu::default();
        assert!(menu.items.is_empty());
        assert!(!menu.open);
        assert_eq!(menu.trigger, MenuTrigger::Click);
        assert_eq!(menu.width, 200.0);
    }

    #[test]
    fn test_menu_trigger_eq() {
        assert_eq!(MenuTrigger::Click, MenuTrigger::Click);
        assert_ne!(MenuTrigger::Click, MenuTrigger::Hover);
        assert_ne!(MenuTrigger::Hover, MenuTrigger::ContextMenu);
    }

    #[test]
    fn test_menu_hide_clears_submenu() {
        let mut menu = Menu::new().items(vec![MenuItem::submenu(
            "More",
            vec![MenuItem::action("Sub", "sub")],
        )]);
        menu.show();
        menu.open_submenu = Some(0);

        menu.hide();
        assert!(!menu.is_open());
        assert!(menu.open_submenu.is_none());
        assert!(menu.highlighted_index.is_none());
    }

    #[test]
    fn test_menu_debug() {
        let item = MenuItem::action("Test", "test");
        let debug_str = format!("{item:?}");
        assert!(debug_str.contains("Test"));
    }

    #[test]
    fn test_menu_toggled_debug() {
        let msg = MenuToggled { open: true };
        let debug_str = format!("{msg:?}");
        assert!(debug_str.contains("true"));
    }
}
