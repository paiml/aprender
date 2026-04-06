//! Mock DOM for WASM Testing
//!
//! This module provides DOM abstractions that enable 100% test coverage
//! without requiring actual browser/web-sys dependencies.
//!
//! Probar: Visual feedback - Visual management through observable DOM state

use std::collections::HashMap;

/// Represents a DOM element for testing
#[derive(Debug, Clone, PartialEq)]
pub struct DomElement {
    /// Element ID
    pub id: String,
    /// Element tag name
    pub tag: String,
    /// Text content
    pub text_content: String,
    /// Element attributes
    pub attributes: HashMap<String, String>,
    /// CSS classes
    pub classes: Vec<String>,
    /// Whether element is visible
    pub visible: bool,
    /// Child elements
    pub children: Vec<DomElement>,
}

impl Default for DomElement {
    fn default() -> Self {
        Self::new("div")
    }
}

impl DomElement {
    /// Creates a new DOM element with the given tag
    #[must_use]
    pub fn new(tag: &str) -> Self {
        Self {
            id: String::new(),
            tag: tag.to_string(),
            text_content: String::new(),
            attributes: HashMap::new(),
            classes: Vec::new(),
            visible: true,
            children: Vec::new(),
        }
    }

    /// Creates an element with an ID
    #[must_use]
    pub fn with_id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    /// Sets the text content
    #[must_use]
    pub fn with_text(mut self, text: &str) -> Self {
        self.text_content = text.to_string();
        self
    }

    /// Adds a class
    #[must_use]
    pub fn with_class(mut self, class: &str) -> Self {
        self.classes.push(class.to_string());
        self
    }

    /// Sets an attribute
    #[must_use]
    pub fn with_attr(mut self, key: &str, value: &str) -> Self {
        self.attributes.insert(key.to_string(), value.to_string());
        self
    }

    /// Adds a child element
    #[must_use]
    pub fn with_child(mut self, child: DomElement) -> Self {
        self.children.push(child);
        self
    }

    /// Sets visibility
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Sets text content
    pub fn set_text(&mut self, text: &str) {
        self.text_content = text.to_string();
    }

    /// Adds a class
    pub fn add_class(&mut self, class: &str) {
        if !self.classes.contains(&class.to_string()) {
            self.classes.push(class.to_string());
        }
    }

    /// Removes a class
    pub fn remove_class(&mut self, class: &str) {
        self.classes.retain(|c| c != class);
    }

    /// Checks if element has a class
    #[must_use]
    pub fn has_class(&self, class: &str) -> bool {
        self.classes.contains(&class.to_string())
    }

    /// Gets an attribute value
    #[must_use]
    pub fn get_attr(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).map(|s| s.as_str())
    }
}

/// DOM events that can be dispatched
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DomEvent {
    /// Click event on an element
    Click {
        /// The ID of the clicked element
        element_id: String,
    },
    /// Input event with new value
    Input {
        /// The ID of the input element
        element_id: String,
        /// The new value entered
        value: String,
    },
    /// Key press event
    KeyPress {
        /// The key that was pressed
        key: String,
        /// Whether Ctrl was held
        ctrl: bool,
        /// Whether Shift was held
        shift: bool,
    },
    /// Focus event on an element
    Focus {
        /// The ID of the focused element
        element_id: String,
    },
    /// Blur event (element lost focus)
    Blur {
        /// The ID of the element that lost focus
        element_id: String,
    },
    /// Submit event (for forms)
    Submit {
        /// The ID of the submitted form
        element_id: String,
    },
}

impl DomEvent {
    /// Creates a click event
    #[must_use]
    pub fn click(element_id: &str) -> Self {
        Self::Click {
            element_id: element_id.to_string(),
        }
    }

    /// Creates an input event
    #[must_use]
    pub fn input(element_id: &str, value: &str) -> Self {
        Self::Input {
            element_id: element_id.to_string(),
            value: value.to_string(),
        }
    }

    /// Creates a key press event
    #[must_use]
    pub fn key_press(key: &str) -> Self {
        Self::KeyPress {
            key: key.to_string(),
            ctrl: false,
            shift: false,
        }
    }

    /// Creates a key press with modifiers
    #[must_use]
    pub fn key_press_with_modifiers(key: &str, ctrl: bool, shift: bool) -> Self {
        Self::KeyPress {
            key: key.to_string(),
            ctrl,
            shift,
        }
    }

    /// Creates a focus event
    #[must_use]
    pub fn focus(element_id: &str) -> Self {
        Self::Focus {
            element_id: element_id.to_string(),
        }
    }

    /// Creates a blur event
    #[must_use]
    pub fn blur(element_id: &str) -> Self {
        Self::Blur {
            element_id: element_id.to_string(),
        }
    }

    /// Creates a submit event
    #[must_use]
    pub fn submit(element_id: &str) -> Self {
        Self::Submit {
            element_id: element_id.to_string(),
        }
    }
}

/// Mock DOM for testing WASM calculator without browser
#[derive(Debug)]
pub struct MockDom {
    /// Root element
    pub root: DomElement,
    /// Elements by ID for quick lookup
    elements: HashMap<String, DomElement>,
    /// Event history for verification
    event_history: Vec<DomEvent>,
    /// Focused element ID
    focused_element: Option<String>,
}

impl Default for MockDom {
    fn default() -> Self {
        Self::new()
    }
}

impl MockDom {
    /// Creates a new mock DOM
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: DomElement::new("div").with_id("root"),
            elements: HashMap::new(),
            event_history: Vec::new(),
            focused_element: None,
        }
    }

    /// Creates a calculator DOM structure
    #[must_use]
    pub fn calculator() -> Self {
        let mut dom = Self::new();

        // Input field
        let input = DomElement::new("input")
            .with_id("calc-input")
            .with_attr("type", "text")
            .with_attr("placeholder", "Enter expression");

        // Result display
        let result = DomElement::new("div")
            .with_id("calc-result")
            .with_class("result-display");

        // History list
        let history = DomElement::new("ul")
            .with_id("calc-history")
            .with_class("history-list");

        // Status display for Anomaly
        let status = DomElement::new("div")
            .with_id("calc-status")
            .with_class("status-display");

        // Buttons
        let clear_btn = DomElement::new("button")
            .with_id("btn-clear")
            .with_text("C");

        let equals_btn = DomElement::new("button")
            .with_id("btn-equals")
            .with_text("=");

        // Build root
        dom.root = DomElement::new("div")
            .with_id("calculator")
            .with_class("calculator-app")
            .with_child(input.clone())
            .with_child(result.clone())
            .with_child(history.clone())
            .with_child(status.clone())
            .with_child(clear_btn.clone())
            .with_child(equals_btn.clone());

        // Register elements
        dom.register_element(input);
        dom.register_element(result);
        dom.register_element(history);
        dom.register_element(status);
        dom.register_element(clear_btn);
        dom.register_element(equals_btn);

        dom
    }

    /// Registers an element for ID lookup
    pub fn register_element(&mut self, element: DomElement) {
        if !element.id.is_empty() {
            self.elements.insert(element.id.clone(), element);
        }
    }

    /// Gets an element by ID
    #[must_use]
    pub fn get_element(&self, id: &str) -> Option<&DomElement> {
        self.elements.get(id)
    }

    /// Gets a mutable element by ID
    pub fn get_element_mut(&mut self, id: &str) -> Option<&mut DomElement> {
        self.elements.get_mut(id)
    }

    /// Dispatches an event
    pub fn dispatch_event(&mut self, event: DomEvent) {
        self.event_history.push(event.clone());

        match &event {
            DomEvent::Focus { element_id } => {
                self.focused_element = Some(element_id.clone());
            }
            DomEvent::Blur { .. } => {
                self.focused_element = None;
            }
            DomEvent::Input { element_id, value } => {
                if let Some(elem) = self.elements.get_mut(element_id) {
                    elem.set_text(value);
                    elem.attributes.insert("value".to_string(), value.clone());
                }
            }
            _ => {}
        }
    }

    /// Gets the event history
    #[must_use]
    pub fn event_history(&self) -> &[DomEvent] {
        &self.event_history
    }

    /// Clears event history
    pub fn clear_event_history(&mut self) {
        self.event_history.clear();
    }

    /// Gets the currently focused element ID
    #[must_use]
    pub fn focused_element(&self) -> Option<&str> {
        self.focused_element.as_deref()
    }

    /// Updates element text by ID
    pub fn set_element_text(&mut self, id: &str, text: &str) {
        if let Some(elem) = self.elements.get_mut(id) {
            elem.set_text(text);
        }
    }

    /// Gets element text by ID
    #[must_use]
    pub fn get_element_text(&self, id: &str) -> Option<&str> {
        self.elements.get(id).map(|e| e.text_content.as_str())
    }

    /// Adds a child element to a parent
    pub fn append_child(&mut self, parent_id: &str, child: DomElement) {
        let child_id = child.id.clone();
        if let Some(parent) = self.elements.get_mut(parent_id) {
            parent.children.push(child.clone());
        }
        if !child_id.is_empty() {
            self.elements.insert(child_id, child);
        }
    }

    /// Clears children of an element
    pub fn clear_children(&mut self, id: &str) {
        // First, collect child IDs to remove
        let child_ids: Vec<String> = self
            .elements
            .get(id)
            .map(|elem| {
                elem.children
                    .iter()
                    .filter(|c| !c.id.is_empty())
                    .map(|c| c.id.clone())
                    .collect()
            })
            .unwrap_or_default();

        // Remove child elements from registry
        for child_id in child_ids {
            self.elements.remove(&child_id);
        }

        // Clear children list
        if let Some(elem) = self.elements.get_mut(id) {
            elem.children.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== DomElement tests =====

    #[test]
    fn test_dom_element_new() {
        let elem = DomElement::new("span");
        assert_eq!(elem.tag, "span");
        assert!(elem.id.is_empty());
        assert!(elem.text_content.is_empty());
    }

    #[test]
    fn test_dom_element_default() {
        let elem = DomElement::default();
        assert_eq!(elem.tag, "div");
    }

    #[test]
    fn test_dom_element_with_id() {
        let elem = DomElement::new("div").with_id("test-id");
        assert_eq!(elem.id, "test-id");
    }

    #[test]
    fn test_dom_element_with_text() {
        let elem = DomElement::new("p").with_text("Hello");
        assert_eq!(elem.text_content, "Hello");
    }

    #[test]
    fn test_dom_element_with_class() {
        let elem = DomElement::new("div").with_class("active");
        assert!(elem.classes.contains(&"active".to_string()));
    }

    #[test]
    fn test_dom_element_with_attr() {
        let elem = DomElement::new("input").with_attr("type", "text");
        assert_eq!(elem.get_attr("type"), Some("text"));
    }

    #[test]
    fn test_dom_element_with_child() {
        let child = DomElement::new("span").with_text("child");
        let parent = DomElement::new("div").with_child(child);
        assert_eq!(parent.children.len(), 1);
        assert_eq!(parent.children[0].text_content, "child");
    }

    #[test]
    fn test_dom_element_set_visible() {
        let mut elem = DomElement::new("div");
        assert!(elem.visible);
        elem.set_visible(false);
        assert!(!elem.visible);
    }

    #[test]
    fn test_dom_element_set_text() {
        let mut elem = DomElement::new("div");
        elem.set_text("Updated");
        assert_eq!(elem.text_content, "Updated");
    }

    #[test]
    fn test_dom_element_add_class() {
        let mut elem = DomElement::new("div");
        elem.add_class("foo");
        elem.add_class("bar");
        elem.add_class("foo"); // duplicate should not be added
        assert_eq!(elem.classes.len(), 2);
    }

    #[test]
    fn test_dom_element_remove_class() {
        let mut elem = DomElement::new("div").with_class("foo").with_class("bar");
        elem.remove_class("foo");
        assert!(!elem.has_class("foo"));
        assert!(elem.has_class("bar"));
    }

    #[test]
    fn test_dom_element_has_class() {
        let elem = DomElement::new("div").with_class("active");
        assert!(elem.has_class("active"));
        assert!(!elem.has_class("inactive"));
    }

    #[test]
    fn test_dom_element_get_attr_none() {
        let elem = DomElement::new("div");
        assert_eq!(elem.get_attr("missing"), None);
    }

    #[test]
    fn test_dom_element_clone() {
        let elem = DomElement::new("div").with_id("test").with_text("content");
        let cloned = elem.clone();
        assert_eq!(elem, cloned);
    }

    #[test]
    fn test_dom_element_debug() {
        let elem = DomElement::new("div");
        let debug = format!("{:?}", elem);
        assert!(debug.contains("DomElement"));
    }

    // ===== DomEvent tests =====

    #[test]
    fn test_dom_event_click() {
        let event = DomEvent::click("btn");
        assert!(matches!(event, DomEvent::Click { element_id } if element_id == "btn"));
    }

    #[test]
    fn test_dom_event_input() {
        let event = DomEvent::input("field", "value");
        assert!(
            matches!(event, DomEvent::Input { element_id, value } if element_id == "field" && value == "value")
        );
    }

    #[test]
    fn test_dom_event_key_press() {
        let event = DomEvent::key_press("Enter");
        assert!(
            matches!(event, DomEvent::KeyPress { key, ctrl, shift } if key == "Enter" && !ctrl && !shift)
        );
    }

    #[test]
    fn test_dom_event_key_press_with_modifiers() {
        let event = DomEvent::key_press_with_modifiers("c", true, false);
        assert!(
            matches!(event, DomEvent::KeyPress { key, ctrl, shift } if key == "c" && ctrl && !shift)
        );
    }

    #[test]
    fn test_dom_event_focus() {
        let event = DomEvent::focus("input");
        assert!(matches!(event, DomEvent::Focus { element_id } if element_id == "input"));
    }

    #[test]
    fn test_dom_event_blur() {
        let event = DomEvent::blur("input");
        assert!(matches!(event, DomEvent::Blur { element_id } if element_id == "input"));
    }

    #[test]
    fn test_dom_event_submit() {
        let event = DomEvent::submit("form");
        assert!(matches!(event, DomEvent::Submit { element_id } if element_id == "form"));
    }

    #[test]
    fn test_dom_event_clone() {
        let event = DomEvent::click("btn");
        let cloned = event.clone();
        assert_eq!(event, cloned);
    }

    #[test]
    fn test_dom_event_debug() {
        let event = DomEvent::click("btn");
        let debug = format!("{:?}", event);
        assert!(debug.contains("Click"));
    }

    // ===== MockDom tests =====

    #[test]
    fn test_mock_dom_new() {
        let dom = MockDom::new();
        assert_eq!(dom.root.id, "root");
        assert!(dom.event_history.is_empty());
    }

    #[test]
    fn test_mock_dom_default() {
        let dom = MockDom::default();
        assert_eq!(dom.root.id, "root");
    }

    #[test]
    fn test_mock_dom_calculator() {
        let dom = MockDom::calculator();
        assert!(dom.get_element("calc-input").is_some());
        assert!(dom.get_element("calc-result").is_some());
        assert!(dom.get_element("calc-history").is_some());
        assert!(dom.get_element("calc-status").is_some());
        assert!(dom.get_element("btn-clear").is_some());
        assert!(dom.get_element("btn-equals").is_some());
    }

    #[test]
    fn test_mock_dom_register_element() {
        let mut dom = MockDom::new();
        let elem = DomElement::new("span").with_id("test");
        dom.register_element(elem);
        assert!(dom.get_element("test").is_some());
    }

    #[test]
    fn test_mock_dom_register_element_no_id() {
        let mut dom = MockDom::new();
        let elem = DomElement::new("span"); // no ID
        let count_before = dom.elements.len();
        dom.register_element(elem);
        assert_eq!(dom.elements.len(), count_before); // should not be registered
    }

    #[test]
    fn test_mock_dom_get_element_mut() {
        let mut dom = MockDom::calculator();
        if let Some(elem) = dom.get_element_mut("calc-result") {
            elem.set_text("42");
        }
        assert_eq!(dom.get_element_text("calc-result"), Some("42"));
    }

    #[test]
    fn test_mock_dom_dispatch_focus() {
        let mut dom = MockDom::calculator();
        dom.dispatch_event(DomEvent::focus("calc-input"));
        assert_eq!(dom.focused_element(), Some("calc-input"));
    }

    #[test]
    fn test_mock_dom_dispatch_blur() {
        let mut dom = MockDom::calculator();
        dom.dispatch_event(DomEvent::focus("calc-input"));
        dom.dispatch_event(DomEvent::blur("calc-input"));
        assert_eq!(dom.focused_element(), None);
    }

    #[test]
    fn test_mock_dom_dispatch_input() {
        let mut dom = MockDom::calculator();
        dom.dispatch_event(DomEvent::input("calc-input", "2 + 2"));
        let elem = dom.get_element("calc-input").unwrap();
        assert_eq!(elem.text_content, "2 + 2");
        assert_eq!(elem.get_attr("value"), Some("2 + 2"));
    }

    #[test]
    fn test_mock_dom_dispatch_click() {
        let mut dom = MockDom::calculator();
        dom.dispatch_event(DomEvent::click("btn-equals"));
        assert_eq!(dom.event_history().len(), 1);
    }

    #[test]
    fn test_mock_dom_event_history() {
        let mut dom = MockDom::calculator();
        dom.dispatch_event(DomEvent::click("btn-clear"));
        dom.dispatch_event(DomEvent::input("calc-input", "5"));
        assert_eq!(dom.event_history().len(), 2);
    }

    #[test]
    fn test_mock_dom_clear_event_history() {
        let mut dom = MockDom::calculator();
        dom.dispatch_event(DomEvent::click("btn-clear"));
        dom.clear_event_history();
        assert!(dom.event_history().is_empty());
    }

    #[test]
    fn test_mock_dom_set_element_text() {
        let mut dom = MockDom::calculator();
        dom.set_element_text("calc-result", "100");
        assert_eq!(dom.get_element_text("calc-result"), Some("100"));
    }

    #[test]
    fn test_mock_dom_get_element_text_none() {
        let dom = MockDom::new();
        assert_eq!(dom.get_element_text("nonexistent"), None);
    }

    #[test]
    fn test_mock_dom_append_child() {
        let mut dom = MockDom::calculator();
        let child = DomElement::new("li")
            .with_id("history-item-1")
            .with_text("2 + 2 = 4");
        dom.append_child("calc-history", child);

        assert!(dom.get_element("history-item-1").is_some());
        let history = dom.get_element("calc-history").unwrap();
        assert_eq!(history.children.len(), 1);
    }

    #[test]
    fn test_mock_dom_append_child_no_id() {
        let mut dom = MockDom::calculator();
        let child = DomElement::new("li").with_text("item");
        let elem_count_before = dom.elements.len();
        dom.append_child("calc-history", child);
        // Child without ID should not be added to elements map
        assert_eq!(dom.elements.len(), elem_count_before);
    }

    #[test]
    fn test_mock_dom_clear_children() {
        let mut dom = MockDom::calculator();
        let child1 = DomElement::new("li").with_id("item1");
        let child2 = DomElement::new("li").with_id("item2");
        dom.append_child("calc-history", child1);
        dom.append_child("calc-history", child2);

        dom.clear_children("calc-history");

        assert!(dom.get_element("item1").is_none());
        assert!(dom.get_element("item2").is_none());
        let history = dom.get_element("calc-history").unwrap();
        assert!(history.children.is_empty());
    }

    #[test]
    fn test_mock_dom_debug() {
        let dom = MockDom::new();
        let debug = format!("{:?}", dom);
        assert!(debug.contains("MockDom"));
    }
}
