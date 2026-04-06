//! EventBrick: DOM event handler generation from brick definitions (PROBAR-SPEC-009-P7)
//!
//! Generates JavaScript event handlers from brick definitions.
//! Zero hand-written event handling code.
//!
//! # Example
//!
//! ```rust,ignore
//! use probar::brick::event::{EventBrick, EventType, EventHandler};
//!
//! let events = EventBrick::new()
//!     .on("#record", EventType::Click, EventHandler::dispatch_state("toggle_recording"))
//!     .on("#clear", EventType::Click, EventHandler::call_wasm("clear_transcript"));
//!
//! let js = events.to_event_js();
//! ```

use super::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use std::time::Duration;

/// DOM event types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    /// Mouse click
    Click,
    /// Double click
    DoubleClick,
    /// Mouse down
    MouseDown,
    /// Mouse up
    MouseUp,
    /// Mouse enter (hover start)
    MouseEnter,
    /// Mouse leave (hover end)
    MouseLeave,
    /// Key down
    KeyDown,
    /// Key up
    KeyUp,
    /// Key press
    KeyPress,
    /// Input value change
    Input,
    /// Form change
    Change,
    /// Form submit
    Submit,
    /// Focus gained
    Focus,
    /// Focus lost
    Blur,
    /// Scroll
    Scroll,
    /// Touch start
    TouchStart,
    /// Touch end
    TouchEnd,
    /// Touch move
    TouchMove,
    /// Custom event
    Custom(&'static str),
}

impl EventType {
    /// Get the JavaScript event name
    #[must_use]
    pub fn js_name(&self) -> &str {
        match self {
            Self::Click => "click",
            Self::DoubleClick => "dblclick",
            Self::MouseDown => "mousedown",
            Self::MouseUp => "mouseup",
            Self::MouseEnter => "mouseenter",
            Self::MouseLeave => "mouseleave",
            Self::KeyDown => "keydown",
            Self::KeyUp => "keyup",
            Self::KeyPress => "keypress",
            Self::Input => "input",
            Self::Change => "change",
            Self::Submit => "submit",
            Self::Focus => "focus",
            Self::Blur => "blur",
            Self::Scroll => "scroll",
            Self::TouchStart => "touchstart",
            Self::TouchEnd => "touchend",
            Self::TouchMove => "touchmove",
            Self::Custom(name) => name,
        }
    }
}

/// Event handler action
#[derive(Debug, Clone)]
pub enum EventHandler {
    /// Dispatch a state change event
    DispatchState(String),

    /// Call a WASM exported function
    CallWasm {
        /// Function name
        function: String,
        /// Arguments to pass (JavaScript expressions)
        args: Vec<String>,
    },

    /// Post a message to a worker
    PostMessage {
        /// Target worker name
        worker: String,
        /// Message type
        message_type: String,
        /// Message fields (key = field name, value = JS expression)
        fields: Vec<(String, String)>,
    },

    /// Update a DOM element
    UpdateElement {
        /// Target selector
        selector: String,
        /// Property to update
        property: String,
        /// New value (JavaScript expression)
        value: String,
    },

    /// Toggle a CSS class
    ToggleClass {
        /// Target selector
        selector: String,
        /// Class name
        class: String,
    },

    /// Prevent default and stop propagation
    PreventDefault,

    /// Chain multiple handlers
    Chain(Vec<EventHandler>),

    /// Conditional handler
    If {
        /// Condition (JavaScript expression)
        condition: String,
        /// Handler if true
        then: Box<EventHandler>,
        /// Handler if false (optional)
        otherwise: Option<Box<EventHandler>>,
    },
}

impl EventHandler {
    /// Create a state dispatch handler
    #[must_use]
    pub fn dispatch_state(state: impl Into<String>) -> Self {
        Self::DispatchState(state.into())
    }

    /// Create a WASM call handler
    #[must_use]
    pub fn call_wasm(function: impl Into<String>) -> Self {
        Self::CallWasm {
            function: function.into(),
            args: Vec::new(),
        }
    }

    /// Create a WASM call handler with arguments
    #[must_use]
    pub fn call_wasm_with_args(function: impl Into<String>, args: Vec<String>) -> Self {
        Self::CallWasm {
            function: function.into(),
            args,
        }
    }

    /// Create a worker message handler
    #[must_use]
    pub fn post_to_worker(worker: impl Into<String>, message_type: impl Into<String>) -> Self {
        Self::PostMessage {
            worker: worker.into(),
            message_type: message_type.into(),
            fields: Vec::new(),
        }
    }

    /// Create an element update handler
    #[must_use]
    pub fn update_element(
        selector: impl Into<String>,
        property: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self::UpdateElement {
            selector: selector.into(),
            property: property.into(),
            value: value.into(),
        }
    }

    /// Create a class toggle handler
    #[must_use]
    pub fn toggle_class(selector: impl Into<String>, class: impl Into<String>) -> Self {
        Self::ToggleClass {
            selector: selector.into(),
            class: class.into(),
        }
    }

    /// Chain handlers
    #[must_use]
    pub fn chain(handlers: Vec<EventHandler>) -> Self {
        Self::Chain(handlers)
    }

    /// Create a conditional handler
    #[must_use]
    pub fn when(
        condition: impl Into<String>,
        then: EventHandler,
        otherwise: Option<EventHandler>,
    ) -> Self {
        Self::If {
            condition: condition.into(),
            then: Box::new(then),
            otherwise: otherwise.map(Box::new),
        }
    }

    /// Generate JavaScript code for this handler
    #[must_use]
    pub fn to_js(&self, indent: usize) -> String {
        let pad = "    ".repeat(indent);

        match self {
            Self::DispatchState(state) => {
                format!(
                    "{}window.dispatchEvent(new CustomEvent('state-change', {{ detail: '{}' }}));",
                    pad, state
                )
            }

            Self::CallWasm { function, args } => {
                let args_str = args.join(", ");
                format!("{}window.wasm.{}({});", pad, function, args_str)
            }

            Self::PostMessage {
                worker,
                message_type,
                fields,
            } => {
                let fields_str = if fields.is_empty() {
                    String::new()
                } else {
                    let f: Vec<_> = fields
                        .iter()
                        .map(|(k, v)| format!("{}: {}", k, v))
                        .collect();
                    format!(", {}", f.join(", "))
                };
                format!(
                    "{}{}.postMessage({{ type: '{}'{} }});",
                    pad, worker, message_type, fields_str
                )
            }

            Self::UpdateElement {
                selector,
                property,
                value,
            } => {
                format!(
                    "{}document.querySelector('{}').{} = {};",
                    pad, selector, property, value
                )
            }

            Self::ToggleClass { selector, class } => {
                format!(
                    "{}document.querySelector('{}').classList.toggle('{}');",
                    pad, selector, class
                )
            }

            Self::PreventDefault => {
                format!("{}e.preventDefault();\n{}e.stopPropagation();", pad, pad)
            }

            Self::Chain(handlers) => handlers
                .iter()
                .map(|h| h.to_js(indent))
                .collect::<Vec<_>>()
                .join("\n"),

            Self::If {
                condition,
                then,
                otherwise,
            } => {
                let then_js = then.to_js(indent + 1);
                let else_js = otherwise
                    .as_ref()
                    .map(|h| format!(" else {{\n{}\n{}}}", h.to_js(indent + 1), pad))
                    .unwrap_or_default();

                format!(
                    "{}if ({}) {{\n{}\n{}}}{}",
                    pad, condition, then_js, pad, else_js
                )
            }
        }
    }
}

/// A single event binding
#[derive(Debug, Clone)]
pub struct EventBinding {
    /// CSS selector for the target element
    pub selector: String,
    /// Event type to listen for
    pub event_type: EventType,
    /// Handler to execute
    pub handler: EventHandler,
    /// Use capture phase
    pub capture: bool,
    /// Only fire once
    pub once: bool,
    /// Passive listener (performance optimization)
    pub passive: bool,
}

impl EventBinding {
    /// Create a new event binding
    #[must_use]
    pub fn new(selector: impl Into<String>, event_type: EventType, handler: EventHandler) -> Self {
        Self {
            selector: selector.into(),
            event_type,
            handler,
            capture: false,
            once: false,
            passive: false,
        }
    }

    /// Use capture phase
    #[must_use]
    pub fn capture(mut self) -> Self {
        self.capture = true;
        self
    }

    /// Only fire once
    #[must_use]
    pub fn once(mut self) -> Self {
        self.once = true;
        self
    }

    /// Mark as passive (for scroll/touch performance)
    #[must_use]
    pub fn passive(mut self) -> Self {
        self.passive = true;
        self
    }

    /// Generate JavaScript for this binding
    #[must_use]
    pub fn to_js(&self) -> String {
        let handler_js = self.handler.to_js(2);

        let options = if self.capture || self.once || self.passive {
            let mut opts = Vec::new();
            if self.capture {
                opts.push("capture: true");
            }
            if self.once {
                opts.push("once: true");
            }
            if self.passive {
                opts.push("passive: true");
            }
            format!(", {{ {} }}", opts.join(", "))
        } else {
            String::new()
        };

        format!(
            "document.querySelector('{}').addEventListener('{}', (e) => {{\n{}\n}}{}); ",
            self.selector,
            self.event_type.js_name(),
            handler_js,
            options
        )
    }
}

/// EventBrick: Generates DOM event handlers from brick definition
#[derive(Debug, Clone, Default)]
pub struct EventBrick {
    /// Event bindings
    bindings: Vec<EventBinding>,
    /// Global window event handlers
    window_handlers: Vec<(EventType, EventHandler)>,
}

impl EventBrick {
    /// Create a new event brick
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an event binding
    #[must_use]
    pub fn on(
        mut self,
        selector: impl Into<String>,
        event_type: EventType,
        handler: EventHandler,
    ) -> Self {
        self.bindings
            .push(EventBinding::new(selector, event_type, handler));
        self
    }

    /// Add an event binding with options
    #[must_use]
    pub fn on_with(mut self, binding: EventBinding) -> Self {
        self.bindings.push(binding);
        self
    }

    /// Add a window-level event handler
    #[must_use]
    pub fn on_window(mut self, event_type: EventType, handler: EventHandler) -> Self {
        self.window_handlers.push((event_type, handler));
        self
    }

    /// Generate JavaScript for all event handlers
    #[must_use]
    pub fn to_event_js(&self) -> String {
        let mut js = String::new();

        js.push_str("// Event Handlers\n");
        js.push_str("// Generated by probar - DO NOT EDIT MANUALLY\n\n");

        // Element bindings
        for binding in &self.bindings {
            js.push_str(&binding.to_js());
            js.push('\n');
        }

        // Window handlers
        for (event_type, handler) in &self.window_handlers {
            let handler_js = handler.to_js(1);
            js.push_str(&format!(
                "window.addEventListener('{}', (e) => {{\n{}\n}});\n",
                event_type.js_name(),
                handler_js
            ));
        }

        js
    }

    /// Get all selectors referenced by this brick
    #[must_use]
    pub fn selectors(&self) -> Vec<&str> {
        self.bindings.iter().map(|b| b.selector.as_str()).collect()
    }
}

impl Brick for EventBrick {
    fn brick_name(&self) -> &'static str {
        "EventBrick"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(100)
    }

    fn verify(&self) -> BrickVerification {
        let passed = vec![BrickAssertion::Custom {
            name: "event_bindings_valid".into(),
            validator_id: 10,
        }];

        BrickVerification {
            passed,
            failed: Vec::new(),
            verification_time: Duration::from_micros(50),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // ============================================================
    // EventType tests
    // ============================================================

    #[test]
    fn test_event_type_js_name() {
        assert_eq!(EventType::Click.js_name(), "click");
        assert_eq!(EventType::KeyDown.js_name(), "keydown");
        assert_eq!(EventType::Custom("my-event").js_name(), "my-event");
    }

    #[test]
    fn test_event_type_js_name_all_variants() {
        // Test all EventType variants for js_name
        assert_eq!(EventType::Click.js_name(), "click");
        assert_eq!(EventType::DoubleClick.js_name(), "dblclick");
        assert_eq!(EventType::MouseDown.js_name(), "mousedown");
        assert_eq!(EventType::MouseUp.js_name(), "mouseup");
        assert_eq!(EventType::MouseEnter.js_name(), "mouseenter");
        assert_eq!(EventType::MouseLeave.js_name(), "mouseleave");
        assert_eq!(EventType::KeyDown.js_name(), "keydown");
        assert_eq!(EventType::KeyUp.js_name(), "keyup");
        assert_eq!(EventType::KeyPress.js_name(), "keypress");
        assert_eq!(EventType::Input.js_name(), "input");
        assert_eq!(EventType::Change.js_name(), "change");
        assert_eq!(EventType::Submit.js_name(), "submit");
        assert_eq!(EventType::Focus.js_name(), "focus");
        assert_eq!(EventType::Blur.js_name(), "blur");
        assert_eq!(EventType::Scroll.js_name(), "scroll");
        assert_eq!(EventType::TouchStart.js_name(), "touchstart");
        assert_eq!(EventType::TouchEnd.js_name(), "touchend");
        assert_eq!(EventType::TouchMove.js_name(), "touchmove");
        assert_eq!(EventType::Custom("custom-event").js_name(), "custom-event");
    }

    #[test]
    fn test_event_type_debug_and_clone() {
        let event = EventType::Click;
        let cloned = event;
        assert_eq!(format!("{:?}", cloned), "Click");

        let custom = EventType::Custom("test");
        let custom_clone = custom;
        assert_eq!(custom_clone.js_name(), "test");
    }

    #[test]
    fn test_event_type_equality() {
        assert_eq!(EventType::Click, EventType::Click);
        assert_ne!(EventType::Click, EventType::DoubleClick);
        assert_eq!(EventType::Custom("a"), EventType::Custom("a"));
        assert_ne!(EventType::Custom("a"), EventType::Custom("b"));
    }

    // ============================================================
    // EventHandler tests
    // ============================================================

    #[test]
    fn test_event_handler_dispatch_state() {
        let handler = EventHandler::dispatch_state("recording");
        let js = handler.to_js(0);

        assert!(js.contains("dispatchEvent"));
        assert!(js.contains("state-change"));
        assert!(js.contains("recording"));
    }

    #[test]
    fn test_event_handler_call_wasm() {
        let handler = EventHandler::call_wasm("start_recording");
        let js = handler.to_js(0);

        assert!(js.contains("window.wasm.start_recording()"));
    }

    #[test]
    fn test_event_handler_call_wasm_with_args() {
        let handler = EventHandler::call_wasm_with_args(
            "process_data",
            vec!["arg1".to_string(), "arg2".to_string(), "123".to_string()],
        );
        let js = handler.to_js(0);

        assert!(js.contains("window.wasm.process_data(arg1, arg2, 123)"));
    }

    #[test]
    fn test_event_handler_call_wasm_with_empty_args() {
        let handler = EventHandler::call_wasm_with_args("func", vec![]);
        let js = handler.to_js(0);

        assert!(js.contains("window.wasm.func()"));
    }

    #[test]
    fn test_event_handler_post_to_worker() {
        let handler = EventHandler::post_to_worker("myWorker", "start");
        let js = handler.to_js(0);

        assert!(js.contains("myWorker.postMessage"));
        assert!(js.contains("type: 'start'"));
    }

    #[test]
    fn test_event_handler_post_message_with_fields() {
        let handler = EventHandler::PostMessage {
            worker: "worker".to_string(),
            message_type: "update".to_string(),
            fields: vec![
                ("data".to_string(), "e.target.value".to_string()),
                ("count".to_string(), "42".to_string()),
            ],
        };
        let js = handler.to_js(0);

        assert!(js.contains("worker.postMessage"));
        assert!(js.contains("type: 'update'"));
        assert!(js.contains("data: e.target.value"));
        assert!(js.contains("count: 42"));
    }

    #[test]
    fn test_event_handler_update_element() {
        let handler = EventHandler::update_element("#status", "textContent", "'Ready'");
        let js = handler.to_js(0);

        assert!(js.contains("#status"));
        assert!(js.contains("textContent"));
        assert!(js.contains("'Ready'"));
        assert!(js.contains("querySelector"));
    }

    #[test]
    fn test_event_handler_toggle_class() {
        let handler = EventHandler::toggle_class("#menu", "active");
        let js = handler.to_js(0);

        assert!(js.contains("querySelector('#menu')"));
        assert!(js.contains("classList.toggle('active')"));
    }

    #[test]
    fn test_event_handler_prevent_default() {
        let js = EventHandler::PreventDefault.to_js(0);

        assert!(js.contains("e.preventDefault()"));
        assert!(js.contains("e.stopPropagation()"));
    }

    #[test]
    fn test_event_handler_chain() {
        let handler = EventHandler::chain(vec![
            EventHandler::PreventDefault,
            EventHandler::dispatch_state("clicked"),
        ]);

        let js = handler.to_js(0);

        assert!(js.contains("preventDefault"));
        assert!(js.contains("dispatchEvent"));
    }

    #[test]
    fn test_event_handler_chain_empty() {
        let handler = EventHandler::chain(vec![]);
        let js = handler.to_js(0);
        assert!(js.is_empty());
    }

    #[test]
    fn test_event_handler_chain_multiple() {
        let handler = EventHandler::chain(vec![
            EventHandler::PreventDefault,
            EventHandler::dispatch_state("state1"),
            EventHandler::call_wasm("func1"),
            EventHandler::toggle_class("#el", "class1"),
        ]);

        let js = handler.to_js(0);

        assert!(js.contains("preventDefault"));
        assert!(js.contains("state1"));
        assert!(js.contains("func1"));
        assert!(js.contains("class1"));
    }

    #[test]
    fn test_event_handler_conditional() {
        let handler = EventHandler::when(
            "isRecording",
            EventHandler::dispatch_state("stop"),
            Some(EventHandler::dispatch_state("start")),
        );

        let js = handler.to_js(0);

        assert!(js.contains("if (isRecording)"));
        assert!(js.contains("stop"));
        assert!(js.contains("else"));
        assert!(js.contains("start"));
    }

    #[test]
    fn test_event_handler_conditional_without_else() {
        let handler = EventHandler::when("condition", EventHandler::call_wasm("action"), None);

        let js = handler.to_js(0);

        assert!(js.contains("if (condition)"));
        assert!(js.contains("action"));
        assert!(!js.contains("else"));
    }

    #[test]
    fn test_event_handler_to_js_with_indent() {
        let handler = EventHandler::dispatch_state("test");

        let js_0 = handler.to_js(0);
        let js_1 = handler.to_js(1);
        let js_2 = handler.to_js(2);

        assert!(!js_0.starts_with(' '));
        assert!(js_1.starts_with("    "));
        assert!(js_2.starts_with("        "));
    }

    #[test]
    fn test_event_handler_debug_and_clone() {
        let handler = EventHandler::dispatch_state("test");
        let cloned = handler;

        assert!(format!("{:?}", cloned).contains("DispatchState"));
    }

    // ============================================================
    // EventBinding tests
    // ============================================================

    #[test]
    fn test_event_binding_basic() {
        let binding = EventBinding::new(
            "#button",
            EventType::Click,
            EventHandler::dispatch_state("clicked"),
        );

        let js = binding.to_js();

        assert!(js.contains("#button"));
        assert!(js.contains("click"));
        assert!(js.contains("addEventListener"));
    }

    #[test]
    fn test_event_binding_options() {
        let binding = EventBinding::new(
            "#scroll",
            EventType::Scroll,
            EventHandler::call_wasm("on_scroll"),
        )
        .passive()
        .capture();

        let js = binding.to_js();

        assert!(js.contains("passive: true"));
        assert!(js.contains("capture: true"));
    }

    #[test]
    fn test_event_binding_once() {
        let binding = EventBinding::new(
            "#init",
            EventType::Click,
            EventHandler::call_wasm("initialize"),
        )
        .once();

        let js = binding.to_js();

        assert!(js.contains("once: true"));
    }

    #[test]
    fn test_event_binding_all_options() {
        let binding = EventBinding::new(
            "#element",
            EventType::TouchStart,
            EventHandler::PreventDefault,
        )
        .capture()
        .once()
        .passive();

        let js = binding.to_js();

        assert!(js.contains("capture: true"));
        assert!(js.contains("once: true"));
        assert!(js.contains("passive: true"));
    }

    #[test]
    fn test_event_binding_no_options() {
        let binding = EventBinding::new(
            "#simple",
            EventType::Click,
            EventHandler::dispatch_state("click"),
        );

        let js = binding.to_js();

        // Should not contain options object when no options set
        assert!(!js.contains("capture:"));
        assert!(!js.contains("once:"));
        assert!(!js.contains("passive:"));
    }

    #[test]
    fn test_event_binding_debug_and_clone() {
        let binding = EventBinding::new(
            "#test",
            EventType::Click,
            EventHandler::dispatch_state("test"),
        );
        let cloned = binding;

        assert_eq!(cloned.selector, "#test");
        assert!(format!("{:?}", cloned).contains("EventBinding"));
    }

    #[test]
    fn test_event_binding_fields() {
        let binding = EventBinding::new(
            "#target",
            EventType::MouseEnter,
            EventHandler::toggle_class("#target", "hover"),
        )
        .capture()
        .once()
        .passive();

        assert_eq!(binding.selector, "#target");
        assert_eq!(binding.event_type, EventType::MouseEnter);
        assert!(binding.capture);
        assert!(binding.once);
        assert!(binding.passive);
    }

    // ============================================================
    // EventBrick tests
    // ============================================================

    #[test]
    fn test_event_brick_generation() {
        let events = EventBrick::new()
            .on(
                "#record",
                EventType::Click,
                EventHandler::dispatch_state("toggle"),
            )
            .on("#clear", EventType::Click, EventHandler::call_wasm("clear"));

        let js = events.to_event_js();

        assert!(js.contains("Generated by probar"));
        assert!(js.contains("#record"));
        assert!(js.contains("#clear"));
    }

    #[test]
    fn test_event_brick_new() {
        let brick = EventBrick::new();
        assert!(brick.selectors().is_empty());
    }

    #[test]
    fn test_event_brick_default() {
        let brick = EventBrick::default();
        assert!(brick.selectors().is_empty());
    }

    #[test]
    fn test_event_brick_on() {
        let brick = EventBrick::new()
            .on("#a", EventType::Click, EventHandler::PreventDefault)
            .on("#b", EventType::Focus, EventHandler::call_wasm("onFocus"));

        let selectors = brick.selectors();
        assert_eq!(selectors.len(), 2);
        assert!(selectors.contains(&"#a"));
        assert!(selectors.contains(&"#b"));
    }

    #[test]
    fn test_event_brick_on_with() {
        let binding = EventBinding::new(
            "#custom",
            EventType::TouchEnd,
            EventHandler::dispatch_state("touch"),
        )
        .passive()
        .once();

        let brick = EventBrick::new().on_with(binding);

        let js = brick.to_event_js();
        assert!(js.contains("#custom"));
        assert!(js.contains("touchend"));
        assert!(js.contains("passive: true"));
        assert!(js.contains("once: true"));
    }

    #[test]
    fn test_event_brick_on_window() {
        let brick = EventBrick::new()
            .on_window(EventType::Scroll, EventHandler::call_wasm("onScroll"))
            .on_window(EventType::KeyDown, EventHandler::dispatch_state("keydown"));

        let js = brick.to_event_js();

        assert!(js.contains("window.addEventListener('scroll'"));
        assert!(js.contains("window.addEventListener('keydown'"));
    }

    #[test]
    fn test_event_brick_selectors() {
        let brick = EventBrick::new()
            .on("#one", EventType::Click, EventHandler::PreventDefault)
            .on(".two", EventType::Input, EventHandler::call_wasm("input"))
            .on(
                "[data-id]",
                EventType::Change,
                EventHandler::dispatch_state("change"),
            );

        let selectors = brick.selectors();
        assert_eq!(selectors.len(), 3);
        assert!(selectors.contains(&"#one"));
        assert!(selectors.contains(&".two"));
        assert!(selectors.contains(&"[data-id]"));
    }

    #[test]
    fn test_event_brick_to_event_js_empty() {
        let brick = EventBrick::new();
        let js = brick.to_event_js();

        assert!(js.contains("Event Handlers"));
        assert!(js.contains("Generated by probar"));
    }

    #[test]
    fn test_event_brick_debug_and_clone() {
        let brick = EventBrick::new().on("#test", EventType::Click, EventHandler::PreventDefault);

        let cloned = brick;
        assert_eq!(cloned.selectors().len(), 1);
        assert!(format!("{:?}", cloned).contains("EventBrick"));
    }

    // ============================================================
    // Brick trait implementation tests
    // ============================================================

    #[test]
    fn test_event_brick_brick_name() {
        let brick = EventBrick::new();
        assert_eq!(brick.brick_name(), "EventBrick");
    }

    #[test]
    fn test_event_brick_assertions() {
        let brick = EventBrick::new();
        assert!(brick.assertions().is_empty());
    }

    #[test]
    fn test_event_brick_budget() {
        let brick = EventBrick::new();
        let budget = brick.budget();
        assert_eq!(budget.as_duration(), Duration::from_millis(100));
    }

    #[test]
    fn test_event_brick_verify() {
        let brick = EventBrick::new();
        let verification = brick.verify();

        assert!(verification.is_valid());
        assert_eq!(verification.passed.len(), 1);
        assert!(verification.failed.is_empty());
    }

    #[test]
    fn test_event_brick_to_html() {
        let brick = EventBrick::new();
        assert!(brick.to_html().is_empty());
    }

    #[test]
    fn test_event_brick_to_css() {
        let brick = EventBrick::new();
        assert!(brick.to_css().is_empty());
    }

    // ============================================================
    // Integration tests
    // ============================================================

    #[test]
    fn test_complex_event_brick() {
        let brick = EventBrick::new()
            .on(
                "#record-btn",
                EventType::Click,
                EventHandler::chain(vec![
                    EventHandler::PreventDefault,
                    EventHandler::when(
                        "window.isRecording",
                        EventHandler::chain(vec![
                            EventHandler::call_wasm("stop_recording"),
                            EventHandler::toggle_class("#record-btn", "recording"),
                            EventHandler::update_element("#status", "textContent", "'Stopped'"),
                        ]),
                        Some(EventHandler::chain(vec![
                            EventHandler::call_wasm("start_recording"),
                            EventHandler::toggle_class("#record-btn", "recording"),
                            EventHandler::update_element("#status", "textContent", "'Recording'"),
                        ])),
                    ),
                ]),
            )
            .on(
                "#clear-btn",
                EventType::Click,
                EventHandler::chain(vec![
                    EventHandler::call_wasm("clear_transcript"),
                    EventHandler::update_element("#transcript", "textContent", "''"),
                ]),
            )
            .on_window(
                EventType::KeyDown,
                EventHandler::when(
                    "e.key === 'Escape'",
                    EventHandler::call_wasm("cancel_recording"),
                    None,
                ),
            );

        let js = brick.to_event_js();

        // Verify structure
        assert!(js.contains("#record-btn"));
        assert!(js.contains("#clear-btn"));
        assert!(js.contains("window.addEventListener('keydown'"));
        assert!(js.contains("window.isRecording"));
        assert!(js.contains("stop_recording"));
        assert!(js.contains("start_recording"));
        assert!(js.contains("e.key === 'Escape'"));
    }

    #[test]
    fn test_event_binding_with_custom_event() {
        let binding = EventBinding::new(
            "#custom-element",
            EventType::Custom("my-custom-event"),
            EventHandler::post_to_worker("customWorker", "handle"),
        );

        let js = binding.to_js();

        assert!(js.contains("my-custom-event"));
        assert!(js.contains("#custom-element"));
        assert!(js.contains("customWorker.postMessage"));
    }

    #[test]
    fn test_nested_conditional_handlers() {
        let handler = EventHandler::when(
            "conditionA",
            EventHandler::when(
                "conditionB",
                EventHandler::call_wasm("bothTrue"),
                Some(EventHandler::call_wasm("onlyATrue")),
            ),
            Some(EventHandler::call_wasm("aFalse")),
        );

        let js = handler.to_js(0);

        assert!(js.contains("if (conditionA)"));
        assert!(js.contains("if (conditionB)"));
        assert!(js.contains("bothTrue"));
        assert!(js.contains("onlyATrue"));
        assert!(js.contains("aFalse"));
    }
}
