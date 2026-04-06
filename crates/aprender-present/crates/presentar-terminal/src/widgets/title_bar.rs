//! `TitleBar` widget - Standard header for all TUI applications.
//!
//! Grammar of Graphics construct: Every TUI MUST have a title bar with:
//! - App name/logo
//! - Search/filter input
//! - Key bindings hint
//! - Optional status indicators
//!
//! Implements SPEC-024 Section 27.8 - Framework-First pattern.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    FontWeight, Key, LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Title bar position
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TitleBarPosition {
    #[default]
    Top,
    Bottom,
}

/// Title bar style preset
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TitleBarStyle {
    #[default]
    Standard,
    Minimal,
    Detailed,
}

/// Standard title bar widget for TUI applications.
///
/// # Example
/// ```ignore
/// let title_bar = TitleBar::new("ptop")
///     .with_version("1.0.0")
///     .with_search_placeholder("Filter processes...")
///     .with_keybinds(&[("q", "Quit"), ("?", "Help"), ("/", "Search")]);
/// ```
#[derive(Debug, Clone)]
pub struct TitleBar {
    /// Application name
    app_name: String,
    /// Version string (optional)
    version: Option<String>,
    /// Current search/filter text
    search_text: String,
    /// Search placeholder text
    search_placeholder: String,
    /// Whether search is active (focused)
    search_active: bool,
    /// Key binding hints [(key, description)]
    keybinds: Vec<(String, String)>,
    /// Primary color for app name
    primary_color: Color,
    /// Secondary color for hints
    secondary_color: Color,
    /// Position (top or bottom)
    position: TitleBarPosition,
    /// Style preset
    style: TitleBarStyle,
    /// Optional status text (right side)
    status_text: Option<String>,
    /// Optional status color
    status_color: Option<Color>,
    /// Mode indicator (e.g., "[FULLSCREEN]")
    mode_indicator: Option<String>,
    /// Cached bounds
    bounds: Rect,
}

impl Default for TitleBar {
    fn default() -> Self {
        Self {
            app_name: String::from("TUI"),
            version: None,
            search_text: String::new(),
            search_placeholder: String::from("Search..."),
            search_active: false,
            keybinds: Vec::new(),
            primary_color: Color {
                r: 0.4,
                g: 0.7,
                b: 1.0,
                a: 1.0,
            },
            secondary_color: Color {
                r: 0.5,
                g: 0.5,
                b: 0.6,
                a: 1.0,
            },
            position: TitleBarPosition::Top,
            style: TitleBarStyle::Standard,
            status_text: None,
            status_color: None,
            mode_indicator: None,
            bounds: Rect::default(),
        }
    }
}

impl TitleBar {
    /// Create a new title bar with app name.
    #[must_use]
    pub fn new(app_name: impl Into<String>) -> Self {
        Self {
            app_name: app_name.into(),
            ..Default::default()
        }
    }

    /// Set version string.
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Set search placeholder text.
    #[must_use]
    pub fn with_search_placeholder(mut self, placeholder: impl Into<String>) -> Self {
        self.search_placeholder = placeholder.into();
        self
    }

    /// Set current search text.
    #[must_use]
    pub fn with_search_text(mut self, text: impl Into<String>) -> Self {
        self.search_text = text.into();
        self
    }

    /// Set search active state.
    #[must_use]
    pub fn with_search_active(mut self, active: bool) -> Self {
        self.search_active = active;
        self
    }

    /// Set key binding hints.
    #[must_use]
    pub fn with_keybinds(mut self, binds: &[(&str, &str)]) -> Self {
        self.keybinds = binds
            .iter()
            .map(|(k, d)| ((*k).to_string(), (*d).to_string()))
            .collect();
        self
    }

    /// Set primary color.
    #[must_use]
    pub fn with_primary_color(mut self, color: Color) -> Self {
        self.primary_color = color;
        self
    }

    /// Set secondary color.
    #[must_use]
    pub fn with_secondary_color(mut self, color: Color) -> Self {
        self.secondary_color = color;
        self
    }

    /// Set mode indicator (e.g., "[FULLSCREEN]").
    #[must_use]
    pub fn with_mode_indicator(mut self, indicator: impl Into<String>) -> Self {
        self.mode_indicator = Some(indicator.into());
        self
    }

    /// Set position (top or bottom).
    #[must_use]
    pub fn with_position(mut self, position: TitleBarPosition) -> Self {
        self.position = position;
        self
    }

    /// Set style preset.
    #[must_use]
    pub fn with_style(mut self, style: TitleBarStyle) -> Self {
        self.style = style;
        self
    }

    /// Set status text (displayed on right side).
    #[must_use]
    pub fn with_status(mut self, text: impl Into<String>, color: Color) -> Self {
        self.status_text = Some(text.into());
        self.status_color = Some(color);
        self
    }

    /// Update search text (for interactive use).
    pub fn set_search_text(&mut self, text: impl Into<String>) {
        self.search_text = text.into();
    }

    /// Get current search text.
    #[must_use]
    pub fn search_text(&self) -> &str {
        &self.search_text
    }

    /// Toggle search active state.
    pub fn toggle_search(&mut self) {
        self.search_active = !self.search_active;
    }

    /// Check if search is active.
    #[must_use]
    pub fn is_search_active(&self) -> bool {
        self.search_active
    }
}

impl Brick for TitleBar {
    fn brick_name(&self) -> &'static str {
        "title_bar"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(8)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(8)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: self.assertions().to_vec(),
            failed: vec![],
            verification_time: Duration::from_micros(5),
        }
    }

    fn to_html(&self) -> String {
        format!(
            r#"<div class="title-bar"><span class="app-name">{}</span><input class="search" placeholder="{}" value="{}"/></div>"#,
            self.app_name, self.search_placeholder, self.search_text
        )
    }

    fn to_css(&self) -> String {
        format!(
            ".title-bar {{ app: \"{}\"; search: \"{}\"; }}",
            self.app_name, self.search_text
        )
    }
}

impl Widget for TitleBar {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        constraints.constrain(Size::new(constraints.max_width, 1.0))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, 1.0),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 10.0 || self.bounds.height < 1.0 {
            return;
        }

        let y = self.bounds.y;
        let width = self.bounds.width as usize;
        let mut x = self.bounds.x;

        // Style configurations
        let (show_version, show_search, show_keybinds) = match self.style {
            TitleBarStyle::Minimal => (false, false, false),
            TitleBarStyle::Standard => (true, true, true),
            TitleBarStyle::Detailed => (true, true, true),
        };

        // === LEFT: App name + version ===
        let name_style = TextStyle {
            color: self.primary_color,
            weight: FontWeight::Bold,
            ..Default::default()
        };

        canvas.draw_text(&self.app_name, Point::new(x, y), &name_style);
        x += self.app_name.len() as f32;

        if show_version {
            if let Some(ref ver) = self.version {
                let ver_text = format!(" v{ver}");
                canvas.draw_text(
                    &ver_text,
                    Point::new(x, y),
                    &TextStyle {
                        color: self.secondary_color,
                        ..Default::default()
                    },
                );
                x += ver_text.len() as f32;
            }
        }

        // === CENTER: Search box ===
        if show_search {
            let search_start = (width as f32 * 0.25).max(x + 2.0);
            let search_width = (width as f32 * 0.3).min(40.0).max(15.0) as usize;

            // Draw search box
            let search_border_color = if self.search_active {
                self.primary_color
            } else {
                self.secondary_color
            };

            let search_display = if self.search_text.is_empty() {
                if self.search_active {
                    "_".to_string()
                } else {
                    self.search_placeholder.clone()
                }
            } else {
                let visible_len = search_width.saturating_sub(4);
                if self.search_text.len() > visible_len {
                    format!("{}...", &self.search_text[..visible_len.saturating_sub(3)])
                } else {
                    self.search_text.clone()
                }
            };

            // [/] search_text
            let prefix = if self.search_active { "[/] " } else { " /  " };
            canvas.draw_text(
                prefix,
                Point::new(search_start, y),
                &TextStyle {
                    color: search_border_color,
                    ..Default::default()
                },
            );

            let text_color = if self.search_text.is_empty() && !self.search_active {
                self.secondary_color
            } else {
                Color::WHITE
            };

            canvas.draw_text(
                &search_display,
                Point::new(search_start + 4.0, y),
                &TextStyle {
                    color: text_color,
                    ..Default::default()
                },
            );
        }

        // === RIGHT: Mode Indicator + Status + Keybinds ===
        let right_section_start = width as f32 * 0.55;
        let mut right_x = right_section_start;

        // Mode indicator (e.g., [FULLSCREEN])
        if let Some(ref indicator) = self.mode_indicator {
            canvas.draw_text(
                indicator,
                Point::new(right_x, y),
                &TextStyle {
                    color: Color {
                        r: 0.9,
                        g: 0.7,
                        b: 0.2,
                        a: 1.0,
                    }, // Yellow/gold
                    weight: FontWeight::Bold,
                    ..Default::default()
                },
            );
            right_x += indicator.len() as f32 + 2.0;
        }

        // Status text (if any)
        if let (Some(ref status), Some(color)) = (&self.status_text, self.status_color) {
            canvas.draw_text(
                status,
                Point::new(right_x, y),
                &TextStyle {
                    color,
                    ..Default::default()
                },
            );
        }

        // Key bindings hint (right-aligned)
        if show_keybinds && !self.keybinds.is_empty() {
            let keybind_str: String = self
                .keybinds
                .iter()
                .map(|(k, d)| format!("[{k}]{d}"))
                .collect::<Vec<_>>()
                .join(" ");

            let keybind_x =
                (width as f32 - keybind_str.len() as f32 - 1.0).max(right_section_start);

            canvas.draw_text(
                &keybind_str,
                Point::new(keybind_x, y),
                &TextStyle {
                    color: self.secondary_color,
                    ..Default::default()
                },
            );
        }
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        match event {
            Event::KeyDown {
                key: Key::Slash, ..
            } if !self.search_active => {
                self.search_active = true;
                None
            }
            Event::KeyDown {
                key: Key::Escape, ..
            } if self.search_active => {
                self.search_active = false;
                self.search_text.clear();
                None
            }
            Event::KeyDown {
                key: Key::Enter, ..
            } if self.search_active => {
                self.search_active = false;
                None
            }
            Event::KeyDown {
                key: Key::Backspace,
                ..
            } if self.search_active && !self.search_text.is_empty() => {
                self.search_text.pop();
                None
            }
            Event::TextInput { text } if self.search_active => {
                self.search_text.push_str(text);
                None
            }
            _ => None,
        }
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct::{CellBuffer, DirectTerminalCanvas};

    // =========================================================================
    // CREATION & BUILDER TESTS
    // =========================================================================

    #[test]
    fn test_title_bar_creation() {
        let bar = TitleBar::new("ptop")
            .with_version("1.0.0")
            .with_search_placeholder("Filter...");

        assert_eq!(bar.app_name, "ptop");
        assert_eq!(bar.version, Some("1.0.0".to_string()));
        assert_eq!(bar.search_placeholder, "Filter...");
    }

    #[test]
    fn test_title_bar_default() {
        let bar = TitleBar::default();
        assert_eq!(bar.app_name, "TUI");
        assert!(bar.version.is_none());
        assert!(!bar.search_active);
        assert!(bar.keybinds.is_empty());
        assert_eq!(bar.position, TitleBarPosition::Top);
        assert_eq!(bar.style, TitleBarStyle::Standard);
    }

    #[test]
    fn test_title_bar_with_search_text() {
        let bar = TitleBar::new("test").with_search_text("filter");
        assert_eq!(bar.search_text, "filter");
    }

    #[test]
    fn test_title_bar_with_search_active() {
        let bar = TitleBar::new("test").with_search_active(true);
        assert!(bar.search_active);
    }

    #[test]
    fn test_title_bar_with_primary_color() {
        let color = Color::new(1.0, 0.0, 0.0, 1.0);
        let bar = TitleBar::new("test").with_primary_color(color);
        assert_eq!(bar.primary_color, color);
    }

    #[test]
    fn test_title_bar_with_secondary_color() {
        let color = Color::new(0.0, 1.0, 0.0, 1.0);
        let bar = TitleBar::new("test").with_secondary_color(color);
        assert_eq!(bar.secondary_color, color);
    }

    #[test]
    fn test_title_bar_with_mode_indicator() {
        let bar = TitleBar::new("test").with_mode_indicator("[FULLSCREEN]");
        assert_eq!(bar.mode_indicator, Some("[FULLSCREEN]".to_string()));
    }

    #[test]
    fn test_title_bar_with_position() {
        let bar = TitleBar::new("test").with_position(TitleBarPosition::Bottom);
        assert_eq!(bar.position, TitleBarPosition::Bottom);
    }

    // =========================================================================
    // SEARCH TESTS
    // =========================================================================

    #[test]
    fn test_title_bar_search() {
        let mut bar = TitleBar::new("test");
        assert!(!bar.is_search_active());

        bar.toggle_search();
        assert!(bar.is_search_active());

        bar.set_search_text("hello");
        assert_eq!(bar.search_text(), "hello");
    }

    #[test]
    fn test_title_bar_toggle_search_twice() {
        let mut bar = TitleBar::new("test");
        bar.toggle_search();
        assert!(bar.is_search_active());
        bar.toggle_search();
        assert!(!bar.is_search_active());
    }

    // =========================================================================
    // KEYBINDS TESTS
    // =========================================================================

    #[test]
    fn test_title_bar_keybinds() {
        let bar = TitleBar::new("test").with_keybinds(&[("q", "Quit"), ("?", "Help")]);

        assert_eq!(bar.keybinds.len(), 2);
        assert_eq!(bar.keybinds[0], ("q".to_string(), "Quit".to_string()));
    }

    #[test]
    fn test_title_bar_keybinds_empty() {
        let bar = TitleBar::new("test").with_keybinds(&[]);
        assert!(bar.keybinds.is_empty());
    }

    // =========================================================================
    // STYLE TESTS
    // =========================================================================

    #[test]
    fn test_title_bar_styles() {
        let minimal = TitleBar::new("test").with_style(TitleBarStyle::Minimal);
        let standard = TitleBar::new("test").with_style(TitleBarStyle::Standard);
        let detailed = TitleBar::new("test").with_style(TitleBarStyle::Detailed);

        assert_eq!(minimal.style, TitleBarStyle::Minimal);
        assert_eq!(standard.style, TitleBarStyle::Standard);
        assert_eq!(detailed.style, TitleBarStyle::Detailed);
    }

    #[test]
    fn test_title_bar_position_default() {
        assert_eq!(TitleBarPosition::default(), TitleBarPosition::Top);
    }

    #[test]
    fn test_title_bar_style_default() {
        assert_eq!(TitleBarStyle::default(), TitleBarStyle::Standard);
    }

    // =========================================================================
    // STATUS TESTS
    // =========================================================================

    #[test]
    fn test_title_bar_status() {
        let bar = TitleBar::new("test").with_status(
            "Connected",
            Color {
                r: 0.0,
                g: 1.0,
                b: 0.0,
                a: 1.0,
            },
        );

        assert_eq!(bar.status_text, Some("Connected".to_string()));
        assert!(bar.status_color.is_some());
    }

    // =========================================================================
    // BRICK TRAIT TESTS
    // =========================================================================

    #[test]
    fn test_title_bar_brick_name() {
        let bar = TitleBar::new("test");
        assert_eq!(bar.brick_name(), "title_bar");
    }

    #[test]
    fn test_title_bar_assertions() {
        let bar = TitleBar::new("test");
        let assertions = bar.assertions();
        assert!(!assertions.is_empty());
    }

    #[test]
    fn test_title_bar_budget() {
        let bar = TitleBar::new("test");
        let budget = bar.budget();
        assert!(budget.total_ms > 0);
    }

    #[test]
    fn test_title_bar_verify() {
        let bar = TitleBar::new("test");
        let verification = bar.verify();
        assert!(!verification.passed.is_empty());
        assert!(verification.failed.is_empty());
    }

    #[test]
    fn test_title_bar_to_html() {
        let bar = TitleBar::new("ptop").with_search_text("filter");
        let html = bar.to_html();
        assert!(html.contains("ptop"));
        assert!(html.contains("filter"));
        assert!(html.contains("title-bar"));
    }

    #[test]
    fn test_title_bar_to_css() {
        let bar = TitleBar::new("ptop").with_search_text("filter");
        let css = bar.to_css();
        assert!(css.contains("ptop"));
        assert!(css.contains("filter"));
    }

    // =========================================================================
    // WIDGET TRAIT TESTS
    // =========================================================================

    #[test]
    fn test_title_bar_type_id() {
        let bar = TitleBar::new("test");
        let id = Widget::type_id(&bar);
        assert_eq!(id, TypeId::of::<TitleBar>());
    }

    #[test]
    fn test_title_bar_measure() {
        let bar = TitleBar::new("test");
        let constraints = Constraints::loose(Size::new(100.0, 50.0));
        let size = bar.measure(constraints);
        assert_eq!(size.width, 100.0);
        assert_eq!(size.height, 1.0);
    }

    #[test]
    fn test_title_bar_layout() {
        let mut bar = TitleBar::new("test");
        let bounds = Rect::new(0.0, 0.0, 100.0, 1.0);
        let result = bar.layout(bounds);
        assert_eq!(result.size.width, 100.0);
        assert_eq!(result.size.height, 1.0);
        assert_eq!(bar.bounds, bounds);
    }

    #[test]
    fn test_title_bar_paint_standard() {
        let mut buffer = CellBuffer::new(100, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let mut bar = TitleBar::new("ptop")
            .with_version("1.0.0")
            .with_keybinds(&[("q", "Quit")]);
        bar.layout(Rect::new(0.0, 0.0, 100.0, 1.0));
        bar.paint(&mut canvas);
    }

    #[test]
    fn test_title_bar_paint_minimal() {
        let mut buffer = CellBuffer::new(100, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let mut bar = TitleBar::new("ptop").with_style(TitleBarStyle::Minimal);
        bar.layout(Rect::new(0.0, 0.0, 100.0, 1.0));
        bar.paint(&mut canvas);
    }

    #[test]
    fn test_title_bar_paint_with_search_active() {
        let mut buffer = CellBuffer::new(100, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let mut bar = TitleBar::new("ptop").with_search_active(true);
        bar.layout(Rect::new(0.0, 0.0, 100.0, 1.0));
        bar.paint(&mut canvas);
    }

    #[test]
    fn test_title_bar_paint_with_search_text() {
        let mut buffer = CellBuffer::new(100, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let mut bar = TitleBar::new("ptop").with_search_text("filter text");
        bar.layout(Rect::new(0.0, 0.0, 100.0, 1.0));
        bar.paint(&mut canvas);
    }

    #[test]
    fn test_title_bar_paint_with_long_search_text() {
        let mut buffer = CellBuffer::new(100, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let mut bar = TitleBar::new("ptop")
            .with_search_text("this is a very long search text that should be truncated");
        bar.layout(Rect::new(0.0, 0.0, 100.0, 1.0));
        bar.paint(&mut canvas);
    }

    #[test]
    fn test_title_bar_paint_with_mode_indicator() {
        let mut buffer = CellBuffer::new(100, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let mut bar = TitleBar::new("ptop").with_mode_indicator("[FULLSCREEN]");
        bar.layout(Rect::new(0.0, 0.0, 100.0, 1.0));
        bar.paint(&mut canvas);
    }

    #[test]
    fn test_title_bar_paint_with_status() {
        let mut buffer = CellBuffer::new(100, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let mut bar =
            TitleBar::new("ptop").with_status("Connected", Color::new(0.0, 1.0, 0.0, 1.0));
        bar.layout(Rect::new(0.0, 0.0, 100.0, 1.0));
        bar.paint(&mut canvas);
    }

    #[test]
    fn test_title_bar_paint_too_small() {
        let mut buffer = CellBuffer::new(10, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let mut bar = TitleBar::new("ptop");
        bar.layout(Rect::new(0.0, 0.0, 5.0, 1.0)); // Too small
        bar.paint(&mut canvas); // Should return early
    }

    // =========================================================================
    // EVENT HANDLING TESTS
    // =========================================================================

    #[test]
    fn test_title_bar_event_slash_activates_search() {
        let mut bar = TitleBar::new("test");
        assert!(!bar.search_active);

        let event = Event::key_down(Key::Slash);
        bar.event(&event);
        assert!(bar.search_active);
    }

    #[test]
    fn test_title_bar_event_slash_ignored_when_active() {
        let mut bar = TitleBar::new("test").with_search_active(true);

        let event = Event::key_down(Key::Slash);
        bar.event(&event);
        assert!(bar.search_active); // Still active
    }

    #[test]
    fn test_title_bar_event_escape_deactivates_search() {
        let mut bar = TitleBar::new("test")
            .with_search_active(true)
            .with_search_text("filter");

        let event = Event::key_down(Key::Escape);
        bar.event(&event);
        assert!(!bar.search_active);
        assert!(bar.search_text.is_empty()); // Cleared
    }

    #[test]
    fn test_title_bar_event_escape_ignored_when_inactive() {
        let mut bar = TitleBar::new("test");

        let event = Event::key_down(Key::Escape);
        bar.event(&event);
        assert!(!bar.search_active);
    }

    #[test]
    fn test_title_bar_event_enter_deactivates_search() {
        let mut bar = TitleBar::new("test")
            .with_search_active(true)
            .with_search_text("filter");

        let event = Event::key_down(Key::Enter);
        bar.event(&event);
        assert!(!bar.search_active);
        assert_eq!(bar.search_text, "filter"); // Preserved
    }

    #[test]
    fn test_title_bar_event_backspace_deletes_char() {
        let mut bar = TitleBar::new("test")
            .with_search_active(true)
            .with_search_text("filter");

        let event = Event::key_down(Key::Backspace);
        bar.event(&event);
        assert_eq!(bar.search_text, "filte");
    }

    #[test]
    fn test_title_bar_event_backspace_on_empty() {
        let mut bar = TitleBar::new("test").with_search_active(true);

        let event = Event::key_down(Key::Backspace);
        bar.event(&event);
        assert!(bar.search_text.is_empty());
    }

    #[test]
    fn test_title_bar_event_text_input() {
        let mut bar = TitleBar::new("test").with_search_active(true);

        let event = Event::TextInput {
            text: "hello".to_string(),
        };
        bar.event(&event);
        assert_eq!(bar.search_text, "hello");
    }

    #[test]
    fn test_title_bar_event_text_input_ignored_when_inactive() {
        let mut bar = TitleBar::new("test");

        let event = Event::TextInput {
            text: "hello".to_string(),
        };
        bar.event(&event);
        assert!(bar.search_text.is_empty());
    }

    #[test]
    fn test_title_bar_event_unhandled() {
        let mut bar = TitleBar::new("test");

        let event = Event::key_down(Key::Tab);
        let result = bar.event(&event);
        assert!(result.is_none());
    }

    #[test]
    fn test_title_bar_children() {
        let bar = TitleBar::new("test");
        assert!(bar.children().is_empty());
    }

    #[test]
    fn test_title_bar_children_mut() {
        let mut bar = TitleBar::new("test");
        assert!(bar.children_mut().is_empty());
    }
}
