//! Test harness for Presentar applications.
//!
//! Zero external dependencies - pure Rust testing.

use presentar_core::{Event, Key, MouseButton, Rect, Widget};
use std::collections::VecDeque;

use crate::selector::Selector;

/// Test harness for interacting with Presentar widgets.
pub struct Harness {
    /// Root widget being tested
    root: Box<dyn Widget>,
    /// Event queue for simulation
    event_queue: VecDeque<Event>,
    /// Current viewport size
    viewport: Rect,
}

impl Harness {
    /// Create a new harness with a root widget.
    pub fn new(root: impl Widget + 'static) -> Self {
        Self {
            root: Box::new(root),
            event_queue: VecDeque::new(),
            viewport: Rect::new(0.0, 0.0, 1280.0, 720.0),
        }
    }

    /// Set the viewport size.
    #[must_use]
    pub const fn viewport(mut self, width: f32, height: f32) -> Self {
        self.viewport = Rect::new(0.0, 0.0, width, height);
        self
    }

    // === Event Simulation ===

    /// Simulate a click on a widget matching the selector.
    pub fn click(&mut self, selector: &str) -> &mut Self {
        if let Some(bounds) = self.query_bounds(selector) {
            let center = bounds.center();
            self.event_queue
                .push_back(Event::MouseMove { position: center });
            self.event_queue.push_back(Event::MouseDown {
                position: center,
                button: MouseButton::Left,
            });
            self.event_queue.push_back(Event::MouseUp {
                position: center,
                button: MouseButton::Left,
            });
            self.process_events();
        }
        self
    }

    /// Simulate typing text into a widget.
    pub fn type_text(&mut self, selector: &str, text: &str) -> &mut Self {
        if self.query(selector).is_some() {
            // Focus the element
            self.event_queue.push_back(Event::FocusIn);

            // Type each character
            for c in text.chars() {
                self.event_queue.push_back(Event::TextInput {
                    text: c.to_string(),
                });
            }

            self.process_events();
        }
        self
    }

    /// Simulate a key press.
    pub fn press_key(&mut self, key: Key) -> &mut Self {
        self.event_queue.push_back(Event::KeyDown {
            key,
            modifiers: Default::default(),
        });
        self.event_queue.push_back(Event::KeyUp {
            key,
            modifiers: Default::default(),
        });
        self.process_events();
        self
    }

    /// Simulate scrolling.
    pub fn scroll(&mut self, selector: &str, delta: f32) -> &mut Self {
        if self.query(selector).is_some() {
            self.event_queue.push_back(Event::Scroll {
                delta_x: 0.0,
                delta_y: delta,
            });
            self.process_events();
        }
        self
    }

    // === Queries ===

    /// Query for a widget matching the selector.
    #[must_use]
    pub fn query(&self, selector: &str) -> Option<&dyn Widget> {
        let sel = Selector::parse(selector).ok()?;
        self.find_widget(&*self.root, &sel)
    }

    /// Query for all widgets matching the selector.
    #[must_use]
    pub fn query_all(&self, selector: &str) -> Vec<&dyn Widget> {
        let Ok(sel) = Selector::parse(selector) else {
            return Vec::new();
        };
        let mut results = Vec::new();
        self.find_all_widgets(&*self.root, &sel, &mut results);
        results
    }

    /// Get text content from a widget.
    #[must_use]
    pub fn text(&self, selector: &str) -> String {
        // Simplified - would extract text from Text widgets
        if let Some(widget) = self.query(selector) {
            if let Some(name) = widget.accessible_name() {
                return name.to_string();
            }
        }
        String::new()
    }

    /// Check if a widget exists.
    #[must_use]
    pub fn exists(&self, selector: &str) -> bool {
        self.query(selector).is_some()
    }

    // === Assertions ===

    /// Assert that a widget exists.
    ///
    /// # Panics
    ///
    /// Panics if the widget does not exist.
    pub fn assert_exists(&self, selector: &str) -> &Self {
        assert!(
            self.exists(selector),
            "Expected widget matching '{selector}' to exist"
        );
        self
    }

    /// Assert that a widget does not exist.
    ///
    /// # Panics
    ///
    /// Panics if the widget exists.
    pub fn assert_not_exists(&self, selector: &str) -> &Self {
        assert!(
            !self.exists(selector),
            "Expected widget matching '{selector}' to not exist"
        );
        self
    }

    /// Assert that text matches exactly.
    ///
    /// # Panics
    ///
    /// Panics if the text does not match.
    pub fn assert_text(&self, selector: &str, expected: &str) -> &Self {
        let actual = self.text(selector);
        assert_eq!(
            actual, expected,
            "Expected text '{expected}' but got '{actual}' for '{selector}'"
        );
        self
    }

    /// Assert that text contains a substring.
    ///
    /// # Panics
    ///
    /// Panics if the text does not contain the substring.
    pub fn assert_text_contains(&self, selector: &str, substring: &str) -> &Self {
        let actual = self.text(selector);
        assert!(
            actual.contains(substring),
            "Expected text for '{selector}' to contain '{substring}' but got '{actual}'"
        );
        self
    }

    /// Assert the count of matching widgets.
    ///
    /// # Panics
    ///
    /// Panics if the count does not match.
    pub fn assert_count(&self, selector: &str, expected: usize) -> &Self {
        let actual = self.query_all(selector).len();
        assert_eq!(
            actual, expected,
            "Expected {expected} widgets matching '{selector}' but found {actual}"
        );
        self
    }

    // === Internal ===

    fn process_events(&mut self) {
        while let Some(event) = self.event_queue.pop_front() {
            self.root.event(&event);
        }
    }

    #[allow(unknown_lints)]
    #[allow(clippy::only_used_in_recursion, clippy::self_only_used_in_recursion)]
    fn find_widget<'a>(
        &'a self,
        widget: &'a dyn Widget,
        selector: &Selector,
    ) -> Option<&'a dyn Widget> {
        if selector.matches(widget) {
            return Some(widget);
        }

        for child in widget.children() {
            if let Some(found) = self.find_widget(child.as_ref(), selector) {
                return Some(found);
            }
        }

        None
    }

    #[allow(unknown_lints)]
    #[allow(clippy::only_used_in_recursion, clippy::self_only_used_in_recursion)]
    fn find_all_widgets<'a>(
        &'a self,
        widget: &'a dyn Widget,
        selector: &Selector,
        results: &mut Vec<&'a dyn Widget>,
    ) {
        if selector.matches(widget) {
            results.push(widget);
        }

        for child in widget.children() {
            self.find_all_widgets(child.as_ref(), selector, results);
        }
    }

    fn query_bounds(&self, selector: &str) -> Option<Rect> {
        // Simplified - would return actual widget bounds
        if self.exists(selector) {
            Some(Rect::new(0.0, 0.0, 100.0, 50.0))
        } else {
            None
        }
    }

    /// Advance simulated time.
    pub fn tick(&mut self, _ms: u64) {
        // Would trigger animations, timers, etc.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use presentar_core::{
        widget::LayoutResult, Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas,
        Constraints, Size, TypeId,
    };
    use std::any::Any;
    use std::time::Duration;

    // Mock widget for testing
    struct MockWidget {
        test_id: Option<String>,
        accessible_name: Option<String>,
        children: Vec<Box<dyn Widget>>,
    }

    impl MockWidget {
        fn new() -> Self {
            Self {
                test_id: None,
                accessible_name: None,
                children: Vec::new(),
            }
        }

        fn with_test_id(mut self, id: &str) -> Self {
            self.test_id = Some(id.to_string());
            self
        }

        fn with_name(mut self, name: &str) -> Self {
            self.accessible_name = Some(name.to_string());
            self
        }

        fn with_child(mut self, child: MockWidget) -> Self {
            self.children.push(Box::new(child));
            self
        }
    }

    impl Brick for MockWidget {
        fn brick_name(&self) -> &'static str {
            "MockWidget"
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &[]
        }

        fn budget(&self) -> BrickBudget {
            BrickBudget::uniform(16)
        }

        fn verify(&self) -> BrickVerification {
            BrickVerification {
                passed: vec![],
                failed: vec![],
                verification_time: Duration::from_micros(1),
            }
        }

        fn to_html(&self) -> String {
            String::new()
        }

        fn to_css(&self) -> String {
            String::new()
        }
    }

    impl Widget for MockWidget {
        fn type_id(&self) -> TypeId {
            TypeId::of::<Self>()
        }
        fn measure(&self, c: Constraints) -> Size {
            c.constrain(Size::new(100.0, 50.0))
        }
        fn layout(&mut self, b: Rect) -> LayoutResult {
            LayoutResult { size: b.size() }
        }
        fn paint(&self, _: &mut dyn Canvas) {}
        fn event(&mut self, _: &Event) -> Option<Box<dyn Any + Send>> {
            None
        }
        fn children(&self) -> &[Box<dyn Widget>] {
            &self.children
        }
        fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
            &mut self.children
        }
        fn test_id(&self) -> Option<&str> {
            self.test_id.as_deref()
        }
        fn accessible_name(&self) -> Option<&str> {
            self.accessible_name.as_deref()
        }
    }

    #[test]
    fn test_harness_exists() {
        let widget = MockWidget::new().with_test_id("root");
        let harness = Harness::new(widget);
        assert!(harness.exists("[data-testid='root']"));
        assert!(!harness.exists("[data-testid='nonexistent']"));
    }

    #[test]
    fn test_harness_assert_exists() {
        let widget = MockWidget::new().with_test_id("root");
        let harness = Harness::new(widget);
        harness.assert_exists("[data-testid='root']");
    }

    #[test]
    #[should_panic(expected = "Expected widget matching")]
    fn test_harness_assert_exists_fails() {
        let widget = MockWidget::new();
        let harness = Harness::new(widget);
        harness.assert_exists("[data-testid='missing']");
    }

    #[test]
    fn test_harness_text() {
        let widget = MockWidget::new()
            .with_test_id("greeting")
            .with_name("Hello World");
        let harness = Harness::new(widget);
        assert_eq!(harness.text("[data-testid='greeting']"), "Hello World");
    }

    #[test]
    fn test_harness_query_all() {
        let widget = MockWidget::new()
            .with_test_id("parent")
            .with_child(MockWidget::new().with_test_id("child"))
            .with_child(MockWidget::new().with_test_id("child"));

        let harness = Harness::new(widget);
        let children = harness.query_all("[data-testid='child']");
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_harness_assert_count() {
        let widget = MockWidget::new()
            .with_child(MockWidget::new().with_test_id("item"))
            .with_child(MockWidget::new().with_test_id("item"))
            .with_child(MockWidget::new().with_test_id("item"));

        let harness = Harness::new(widget);
        harness.assert_count("[data-testid='item']", 3);
    }

    // =========================================================================
    // assert_not_exists Tests
    // =========================================================================

    #[test]
    fn test_harness_assert_not_exists() {
        let widget = MockWidget::new().with_test_id("root");
        let harness = Harness::new(widget);
        harness.assert_not_exists("[data-testid='nonexistent']");
    }

    #[test]
    #[should_panic(expected = "Expected widget matching")]
    fn test_harness_assert_not_exists_fails() {
        let widget = MockWidget::new().with_test_id("root");
        let harness = Harness::new(widget);
        harness.assert_not_exists("[data-testid='root']");
    }

    // =========================================================================
    // assert_text Tests
    // =========================================================================

    #[test]
    fn test_harness_assert_text() {
        let widget = MockWidget::new().with_test_id("label").with_name("Welcome");
        let harness = Harness::new(widget);
        harness.assert_text("[data-testid='label']", "Welcome");
    }

    #[test]
    #[should_panic(expected = "Expected text")]
    fn test_harness_assert_text_fails() {
        let widget = MockWidget::new().with_test_id("label").with_name("Hello");
        let harness = Harness::new(widget);
        harness.assert_text("[data-testid='label']", "Goodbye");
    }

    #[test]
    fn test_harness_assert_text_contains() {
        let widget = MockWidget::new()
            .with_test_id("message")
            .with_name("Welcome to the app");
        let harness = Harness::new(widget);
        harness.assert_text_contains("[data-testid='message']", "Welcome");
        harness.assert_text_contains("[data-testid='message']", "app");
    }

    #[test]
    #[should_panic(expected = "Expected text")]
    fn test_harness_assert_text_contains_fails() {
        let widget = MockWidget::new()
            .with_test_id("message")
            .with_name("Hello World");
        let harness = Harness::new(widget);
        harness.assert_text_contains("[data-testid='message']", "Goodbye");
    }

    // =========================================================================
    // Viewport Tests
    // =========================================================================

    #[test]
    fn test_harness_viewport() {
        let widget = MockWidget::new();
        let harness = Harness::new(widget).viewport(1920.0, 1080.0);
        assert_eq!(harness.viewport.width, 1920.0);
        assert_eq!(harness.viewport.height, 1080.0);
    }

    #[test]
    fn test_harness_default_viewport() {
        let widget = MockWidget::new();
        let harness = Harness::new(widget);
        assert_eq!(harness.viewport.width, 1280.0);
        assert_eq!(harness.viewport.height, 720.0);
    }

    // =========================================================================
    // Event Simulation Tests
    // =========================================================================

    #[test]
    fn test_harness_click() {
        let widget = MockWidget::new().with_test_id("button");
        let mut harness = Harness::new(widget);
        // Should not panic
        harness.click("[data-testid='button']");
    }

    #[test]
    fn test_harness_click_nonexistent() {
        let widget = MockWidget::new();
        let mut harness = Harness::new(widget);
        // Should not panic for nonexistent widget
        harness.click("[data-testid='nonexistent']");
    }

    #[test]
    fn test_harness_type_text() {
        let widget = MockWidget::new().with_test_id("input");
        let mut harness = Harness::new(widget);
        // Should not panic
        harness.type_text("[data-testid='input']", "Hello World");
    }

    #[test]
    fn test_harness_type_text_nonexistent() {
        let widget = MockWidget::new();
        let mut harness = Harness::new(widget);
        // Should not panic for nonexistent widget
        harness.type_text("[data-testid='nonexistent']", "Hello");
    }

    #[test]
    fn test_harness_press_key() {
        let widget = MockWidget::new();
        let mut harness = Harness::new(widget);
        // Should not panic
        harness.press_key(Key::Enter);
        harness.press_key(Key::Escape);
        harness.press_key(Key::Tab);
    }

    #[test]
    fn test_harness_scroll() {
        let widget = MockWidget::new().with_test_id("list");
        let mut harness = Harness::new(widget);
        // Should not panic
        harness.scroll("[data-testid='list']", 100.0);
        harness.scroll("[data-testid='list']", -50.0);
    }

    #[test]
    fn test_harness_scroll_nonexistent() {
        let widget = MockWidget::new();
        let mut harness = Harness::new(widget);
        // Should not panic for nonexistent widget
        harness.scroll("[data-testid='nonexistent']", 100.0);
    }

    // =========================================================================
    // Query Tests
    // =========================================================================

    #[test]
    fn test_harness_query_returns_widget() {
        let widget = MockWidget::new().with_test_id("root").with_name("Root");
        let harness = Harness::new(widget);
        let result = harness.query("[data-testid='root']");
        assert!(result.is_some());
        assert_eq!(result.unwrap().accessible_name(), Some("Root"));
    }

    #[test]
    fn test_harness_query_returns_none() {
        let widget = MockWidget::new();
        let harness = Harness::new(widget);
        let result = harness.query("[data-testid='missing']");
        assert!(result.is_none());
    }

    #[test]
    fn test_harness_query_nested() {
        let widget = MockWidget::new().with_child(
            MockWidget::new()
                .with_test_id("nested")
                .with_name("Nested Widget"),
        );
        let harness = Harness::new(widget);
        let result = harness.query("[data-testid='nested']");
        assert!(result.is_some());
        assert_eq!(result.unwrap().accessible_name(), Some("Nested Widget"));
    }

    #[test]
    fn test_harness_query_all_empty() {
        let widget = MockWidget::new();
        let harness = Harness::new(widget);
        let results = harness.query_all("[data-testid='missing']");
        assert!(results.is_empty());
    }

    #[test]
    fn test_harness_query_all_nested() {
        let widget = MockWidget::new()
            .with_child(
                MockWidget::new()
                    .with_test_id("item")
                    .with_child(MockWidget::new().with_test_id("item")),
            )
            .with_child(MockWidget::new().with_test_id("item"));

        let harness = Harness::new(widget);
        let results = harness.query_all("[data-testid='item']");
        assert_eq!(results.len(), 3);
    }

    // =========================================================================
    // Tick Test
    // =========================================================================

    #[test]
    fn test_harness_tick() {
        let widget = MockWidget::new();
        let mut harness = Harness::new(widget);
        // Should not panic
        harness.tick(100);
        harness.tick(1000);
    }

    // =========================================================================
    // Text Edge Cases
    // =========================================================================

    #[test]
    fn test_harness_text_empty() {
        let widget = MockWidget::new().with_test_id("empty");
        let harness = Harness::new(widget);
        assert_eq!(harness.text("[data-testid='empty']"), "");
    }

    #[test]
    fn test_harness_text_nonexistent() {
        let widget = MockWidget::new();
        let harness = Harness::new(widget);
        assert_eq!(harness.text("[data-testid='missing']"), "");
    }

    // =========================================================================
    // Method Chaining Tests
    // =========================================================================

    #[test]
    fn test_harness_method_chaining() {
        let widget = MockWidget::new()
            .with_test_id("form")
            .with_child(MockWidget::new().with_test_id("input"))
            .with_child(MockWidget::new().with_test_id("submit"));

        let mut harness = Harness::new(widget);

        // Chain multiple operations
        harness
            .click("[data-testid='input']")
            .type_text("[data-testid='input']", "user@example.com")
            .press_key(Key::Tab)
            .click("[data-testid='submit']");

        // Assertions also chain
        harness
            .assert_exists("[data-testid='form']")
            .assert_exists("[data-testid='input']")
            .assert_exists("[data-testid='submit']");
    }

    // =========================================================================
    // assert_count Edge Cases
    // =========================================================================

    #[test]
    fn test_harness_assert_count_zero() {
        let widget = MockWidget::new();
        let harness = Harness::new(widget);
        harness.assert_count("[data-testid='missing']", 0);
    }

    #[test]
    #[should_panic(expected = "Expected")]
    fn test_harness_assert_count_fails() {
        let widget = MockWidget::new()
            .with_child(MockWidget::new().with_test_id("item"))
            .with_child(MockWidget::new().with_test_id("item"));

        let harness = Harness::new(widget);
        harness.assert_count("[data-testid='item']", 5);
    }
}
