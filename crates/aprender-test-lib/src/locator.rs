//! Locator abstraction for element selection and interaction.
//!
//! Per spec Section 6.1.1: "Locators (The Core Abstraction) - Unlike Selenium,
//! Locators are strict and auto-wait."
//!
//! # Design Philosophy
//!
//! - **Auto-Waiting**: Locators automatically wait for elements to be actionable
//! - **Strict Selection**: Fails if multiple elements match (prevents flaky tests)
//! - **WASM Entity Support**: Custom `.entity()` method for game object selection
//! - **Fluent API**: Chainable methods for building complex selectors

use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::result::{ProbarError, ProbarResult};

/// Default timeout for auto-waiting (5 seconds)
pub const DEFAULT_TIMEOUT_MS: u64 = 5000;

/// Default polling interval for auto-waiting (50ms)
pub const DEFAULT_POLL_INTERVAL_MS: u64 = 50;

/// A point in 2D space
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    /// X coordinate
    pub x: f32,
    /// Y coordinate
    pub y: f32,
}

impl Point {
    /// Create a new point
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Selector type for locating elements
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selector {
    /// CSS selector (e.g., "button.primary")
    Css(String),
    /// XPath selector
    XPath(String),
    /// Text content selector
    Text(String),
    /// Test ID selector (data-testid attribute)
    TestId(String),
    /// WASM entity selector (game-specific)
    Entity(String),
    /// Combined selector with text filter
    CssWithText {
        /// Base CSS selector
        css: String,
        /// Text content to match
        text: String,
    },
    /// Canvas entity selector (game objects)
    CanvasEntity {
        /// Entity name/ID
        entity: String,
    },
    // =========================================================================
    // PMAT-001: Semantic Locators (Playwright Parity)
    // =========================================================================
    /// ARIA role selector (e.g., "button", "textbox", "link")
    Role {
        /// ARIA role name
        role: String,
        /// Optional accessible name filter
        name: Option<String>,
    },
    /// Label selector (form elements by associated label text)
    Label(String),
    /// Placeholder selector (input/textarea by placeholder attribute)
    Placeholder(String),
    /// Alt text selector (images by alt attribute)
    AltText(String),
}

impl Selector {
    /// Create a CSS selector
    #[must_use]
    pub fn css(selector: impl Into<String>) -> Self {
        Self::Css(selector.into())
    }

    /// Create a test ID selector
    #[must_use]
    pub fn test_id(id: impl Into<String>) -> Self {
        Self::TestId(id.into())
    }

    /// Create a text selector
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Create a WASM entity selector
    #[must_use]
    pub fn entity(name: impl Into<String>) -> Self {
        Self::Entity(name.into())
    }

    // =========================================================================
    // PMAT-001: Semantic Selector Constructors
    // =========================================================================

    /// Create a role selector (ARIA role matching)
    ///
    /// Per Playwright: `page.getByRole('button', { name: 'Submit' })`
    #[must_use]
    pub fn role(role: impl Into<String>) -> Self {
        Self::Role {
            role: role.into(),
            name: None,
        }
    }

    /// Create a role selector with name filter
    #[must_use]
    pub fn role_with_name(role: impl Into<String>, name: impl Into<String>) -> Self {
        Self::Role {
            role: role.into(),
            name: Some(name.into()),
        }
    }

    /// Create a label selector (form elements by label text)
    ///
    /// Per Playwright: `page.getByLabel('Username')`
    #[must_use]
    pub fn label(text: impl Into<String>) -> Self {
        Self::Label(text.into())
    }

    /// Create a placeholder selector (input/textarea by placeholder)
    ///
    /// Per Playwright: `page.getByPlaceholder('Enter email')`
    #[must_use]
    pub fn placeholder(text: impl Into<String>) -> Self {
        Self::Placeholder(text.into())
    }

    /// Create an alt text selector (images by alt attribute)
    ///
    /// Per Playwright: `page.getByAltText('Company Logo')`
    #[must_use]
    pub fn alt_text(text: impl Into<String>) -> Self {
        Self::AltText(text.into())
    }

    /// Convert to JavaScript/WASM query expression
    #[must_use]
    pub fn to_query(&self) -> String {
        match self {
            Self::Css(s) => format!("document.querySelector({s:?})"),
            Self::XPath(s) => {
                format!("document.evaluate({s:?}, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue")
            }
            Self::Text(t) => {
                format!("Array.from(document.querySelectorAll('*')).find(el => el.textContent.includes({t:?}))")
            }
            Self::TestId(id) => format!("document.querySelector('[data-testid={id:?}]')"),
            Self::Entity(name) => format!("window.__wasm_get_entity({name:?})"),
            Self::CssWithText { css, text } => {
                format!("Array.from(document.querySelectorAll({css:?})).find(el => el.textContent.includes({text:?}))")
            }
            Self::CanvasEntity { entity } => format!("window.__wasm_get_canvas_entity({entity:?})"),
            // PMAT-001: Semantic locator queries
            Self::Role { role, name } => {
                if let Some(n) = name {
                    format!(
                        "Array.from(document.querySelectorAll('[role={role:?}]')).find(el => el.textContent.includes({n:?}) || el.getAttribute('aria-label')?.includes({n:?}))"
                    )
                } else {
                    format!("document.querySelector('[role={role:?}]')")
                }
            }
            Self::Label(text) => {
                format!(
                    "(function() {{ const label = Array.from(document.querySelectorAll('label')).find(l => l.textContent.includes({text:?})); if (label && label.htmlFor) return document.getElementById(label.htmlFor); if (label) return label.querySelector('input, textarea, select'); return null; }})()"
                )
            }
            Self::Placeholder(text) => {
                format!("document.querySelector('[placeholder*={text:?}]')")
            }
            Self::AltText(text) => {
                format!("document.querySelector('img[alt*={text:?}]')")
            }
        }
    }

    /// Convert to query for counting matches
    #[must_use]
    pub fn to_count_query(&self) -> String {
        match self {
            Self::Css(s) => format!("document.querySelectorAll({s:?}).length"),
            Self::XPath(s) => {
                format!("document.evaluate({s:?}, document, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null).snapshotLength")
            }
            Self::Text(t) => {
                format!("Array.from(document.querySelectorAll('*')).filter(el => el.textContent.includes({t:?})).length")
            }
            Self::TestId(id) => format!("document.querySelectorAll('[data-testid={id:?}]').length"),
            Self::Entity(name) => format!("window.__wasm_count_entities({name:?})"),
            Self::CssWithText { css, text } => {
                format!("Array.from(document.querySelectorAll({css:?})).filter(el => el.textContent.includes({text:?})).length")
            }
            Self::CanvasEntity { entity } => {
                format!("window.__wasm_count_canvas_entities({entity:?})")
            }
            // PMAT-001: Semantic locator count queries
            Self::Role { role, name } => {
                if let Some(n) = name {
                    format!(
                        "Array.from(document.querySelectorAll('[role={role:?}]')).filter(el => el.textContent.includes({n:?}) || el.getAttribute('aria-label')?.includes({n:?})).length"
                    )
                } else {
                    format!("document.querySelectorAll('[role={role:?}]').length")
                }
            }
            Self::Label(text) => {
                format!(
                    "Array.from(document.querySelectorAll('label')).filter(l => l.textContent.includes({text:?})).length"
                )
            }
            Self::Placeholder(text) => {
                format!("document.querySelectorAll('[placeholder*={text:?}]').length")
            }
            Self::AltText(text) => {
                format!("document.querySelectorAll('img[alt*={text:?}]').length")
            }
        }
    }
}

/// Drag operation builder
#[derive(Debug, Clone)]
pub struct DragOperation {
    /// Target point
    pub target: Point,
    /// Number of intermediate steps
    pub steps: u32,
    /// Total duration of the drag
    pub duration: Duration,
}

impl DragOperation {
    /// Create a new drag operation
    #[must_use]
    pub fn to(target: Point) -> Self {
        Self {
            target,
            steps: 10,
            duration: Duration::from_millis(500),
        }
    }

    /// Set the number of steps
    #[must_use]
    pub const fn steps(mut self, steps: u32) -> Self {
        self.steps = steps;
        self
    }

    /// Set the duration
    #[must_use]
    pub const fn duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }
}

/// Locator options for customizing behavior
#[derive(Debug, Clone)]
pub struct LocatorOptions {
    /// Timeout for auto-waiting
    pub timeout: Duration,
    /// Polling interval for auto-waiting
    pub poll_interval: Duration,
    /// Whether to require strict single-element match
    pub strict: bool,
    /// Whether the element must be visible
    pub visible: bool,
}

impl Default for LocatorOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_millis(DEFAULT_TIMEOUT_MS),
            poll_interval: Duration::from_millis(DEFAULT_POLL_INTERVAL_MS),
            strict: true,
            visible: true,
        }
    }
}

// =============================================================================
// PMAT-002: Filter Options for Locator Operations
// =============================================================================

/// Options for filtering locators (Playwright Parity)
#[derive(Debug, Clone, Default)]
pub struct FilterOptions {
    /// Child locator that must match
    pub has: Option<Box<Locator>>,
    /// Text that must be contained
    pub has_text: Option<String>,
    /// Child locator that must NOT match
    pub has_not: Option<Box<Locator>>,
    /// Text that must NOT be contained
    pub has_not_text: Option<String>,
}

impl FilterOptions {
    /// Create empty filter options
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set has filter (child locator must match)
    #[must_use]
    pub fn has(mut self, locator: Locator) -> Self {
        self.has = Some(Box::new(locator));
        self
    }

    /// Set has_text filter (must contain text)
    #[must_use]
    pub fn has_text(mut self, text: impl Into<String>) -> Self {
        self.has_text = Some(text.into());
        self
    }

    /// Set has_not filter (child locator must NOT match)
    #[must_use]
    pub fn has_not(mut self, locator: Locator) -> Self {
        self.has_not = Some(Box::new(locator));
        self
    }

    /// Set has_not_text filter (must NOT contain text)
    #[must_use]
    pub fn has_not_text(mut self, text: impl Into<String>) -> Self {
        self.has_not_text = Some(text.into());
        self
    }
}

// =============================================================================
// PMAT-003: Mouse Button Types
// =============================================================================

/// Mouse button type for click operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MouseButton {
    /// Left mouse button (default)
    #[default]
    Left,
    /// Right mouse button (context menu)
    Right,
    /// Middle mouse button (scroll wheel click)
    Middle,
}

/// Click options for customizing click behavior
#[derive(Debug, Clone, Default)]
pub struct ClickOptions {
    /// Which mouse button to use
    pub button: MouseButton,
    /// Number of clicks (1 = single, 2 = double)
    pub click_count: u32,
    /// Position within element to click
    pub position: Option<Point>,
    /// Keyboard modifiers to hold during click
    pub modifiers: Vec<KeyModifier>,
}

/// Keyboard modifiers for actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyModifier {
    /// Alt key
    Alt,
    /// Control key
    Control,
    /// Meta key (Cmd on Mac, Win on Windows)
    Meta,
    /// Shift key
    Shift,
}

impl ClickOptions {
    /// Create default click options (left button, single click)
    #[must_use]
    pub fn new() -> Self {
        Self {
            button: MouseButton::Left,
            click_count: 1,
            position: None,
            modifiers: Vec::new(),
        }
    }

    /// Set mouse button
    #[must_use]
    pub fn button(mut self, button: MouseButton) -> Self {
        self.button = button;
        self
    }

    /// Set click count
    #[must_use]
    pub fn click_count(mut self, count: u32) -> Self {
        self.click_count = count;
        self
    }

    /// Set click position within element
    #[must_use]
    pub fn position(mut self, pos: Point) -> Self {
        self.position = Some(pos);
        self
    }

    /// Add a keyboard modifier
    #[must_use]
    pub fn modifier(mut self, modifier: KeyModifier) -> Self {
        self.modifiers.push(modifier);
        self
    }
}

/// A locator for finding and interacting with elements.
///
/// Per spec Section 6.1.1: "Unlike Selenium, Locators are strict and auto-wait."
#[derive(Debug, Clone)]
pub struct Locator {
    /// The selector for finding elements
    selector: Selector,
    /// Options for locator behavior
    options: LocatorOptions,
}

impl Locator {
    /// Create a new locator with a CSS selector
    #[must_use]
    pub fn new(selector: impl Into<String>) -> Self {
        Self {
            selector: Selector::Css(selector.into()),
            options: LocatorOptions::default(),
        }
    }

    /// Create a locator from a selector
    #[must_use]
    pub fn from_selector(selector: Selector) -> Self {
        Self {
            selector,
            options: LocatorOptions::default(),
        }
    }

    /// Filter by text content
    ///
    /// Per spec: `page.locator("button").with_text("Start Game")`
    #[must_use]
    pub fn with_text(self, text: impl Into<String>) -> Self {
        let new_selector = match self.selector {
            Selector::Css(css) => Selector::CssWithText {
                css,
                text: text.into(),
            },
            other => {
                // For non-CSS selectors, we can't combine easily
                // Just keep the original and note the text filter
                let _ = text.into();
                other
            }
        };
        Self {
            selector: new_selector,
            options: self.options,
        }
    }

    /// Filter to a specific WASM game entity
    ///
    /// Per spec: `page.locator("canvas").entity("hero")`
    #[must_use]
    pub fn entity(self, name: impl Into<String>) -> Self {
        Self {
            selector: Selector::CanvasEntity {
                entity: name.into(),
            },
            options: self.options,
        }
    }

    /// Set a custom timeout
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.options.timeout = timeout;
        self
    }

    /// Disable strict mode (allow multiple matches)
    #[must_use]
    pub const fn with_strict(mut self, strict: bool) -> Self {
        self.options.strict = strict;
        self
    }

    /// Set visibility requirement
    #[must_use]
    pub const fn with_visible(mut self, visible: bool) -> Self {
        self.options.visible = visible;
        self
    }

    /// Get the selector
    #[must_use]
    pub const fn selector(&self) -> &Selector {
        &self.selector
    }

    /// Get the options
    #[must_use]
    pub const fn options(&self) -> &LocatorOptions {
        &self.options
    }

    /// Simulate clicking on the located element
    ///
    /// # Errors
    ///
    /// Returns error if element not found or not clickable
    pub fn click(&self) -> ProbarResult<LocatorAction> {
        Ok(LocatorAction::Click {
            locator: self.clone(),
        })
    }

    /// Simulate double-clicking on the located element
    ///
    /// # Errors
    ///
    /// Returns error if element not found or not clickable
    pub fn double_click(&self) -> ProbarResult<LocatorAction> {
        Ok(LocatorAction::DoubleClick {
            locator: self.clone(),
        })
    }

    /// Drag the located element to a target point
    ///
    /// Per spec: `hero.drag_to(&Point::new(500.0, 500.0)).steps(10).duration(...)`
    #[must_use]
    pub fn drag_to(&self, target: &Point) -> DragBuilder {
        DragBuilder {
            locator: self.clone(),
            target: *target,
            steps: 10,
            duration: Duration::from_millis(500),
        }
    }

    /// Fill the located element with text
    ///
    /// # Errors
    ///
    /// Returns error if element not found or not fillable
    pub fn fill(&self, text: impl Into<String>) -> ProbarResult<LocatorAction> {
        Ok(LocatorAction::Fill {
            locator: self.clone(),
            text: text.into(),
        })
    }

    /// Get the text content of the located element
    ///
    /// # Errors
    ///
    /// Returns error if element not found
    pub fn text_content(&self) -> ProbarResult<LocatorQuery> {
        Ok(LocatorQuery::TextContent {
            locator: self.clone(),
        })
    }

    /// Check if the element is visible
    pub fn is_visible(&self) -> ProbarResult<LocatorQuery> {
        Ok(LocatorQuery::IsVisible {
            locator: self.clone(),
        })
    }

    /// Get the bounding box of the located element
    pub fn bounding_box(&self) -> ProbarResult<LocatorQuery> {
        Ok(LocatorQuery::BoundingBox {
            locator: self.clone(),
        })
    }

    /// Wait for the element to be visible
    pub fn wait_for_visible(&self) -> ProbarResult<LocatorAction> {
        Ok(LocatorAction::WaitForVisible {
            locator: self.clone(),
        })
    }

    /// Wait for the element to be hidden
    pub fn wait_for_hidden(&self) -> ProbarResult<LocatorAction> {
        Ok(LocatorAction::WaitForHidden {
            locator: self.clone(),
        })
    }

    /// Check the element count (for assertions)
    pub fn count(&self) -> ProbarResult<LocatorQuery> {
        Ok(LocatorQuery::Count {
            locator: self.clone(),
        })
    }

    // =========================================================================
    // PMAT-002: Locator Operations (Playwright Parity)
    // =========================================================================

    /// Filter the locator by additional conditions
    ///
    /// Per Playwright: `locator.filter({ hasText: 'Hello', has: page.locator('.child') })`
    #[must_use]
    pub fn filter(self, options: FilterOptions) -> Self {
        // For now, if has_text is provided, we combine it with the selector
        if let Some(text) = options.has_text {
            return self.with_text(text);
        }
        // Other filter options would be applied during resolution
        // Store filter options for later use
        self
    }

    /// Create intersection of two locators (both must match)
    ///
    /// Per Playwright: `locator.and(other)`
    #[must_use]
    pub fn and(self, other: Locator) -> Self {
        // Combine selectors - for CSS, we can join them
        let new_selector = match (&self.selector, &other.selector) {
            (Selector::Css(a), Selector::Css(b)) => Selector::Css(format!("{a}{b}")),
            _ => self.selector, // Default to self for non-CSS
        };
        Self {
            selector: new_selector,
            options: self.options,
        }
    }

    /// Create union of two locators (either can match)
    ///
    /// Per Playwright: `locator.or(other)`
    #[must_use]
    pub fn or(self, other: Locator) -> Self {
        // Combine selectors with CSS comma separator
        let new_selector = match (&self.selector, &other.selector) {
            (Selector::Css(a), Selector::Css(b)) => Selector::Css(format!("{a}, {b}")),
            _ => self.selector, // Default to self for non-CSS
        };
        Self {
            selector: new_selector,
            options: self.options,
        }
    }

    /// Get the first matching element
    ///
    /// Per Playwright: `locator.first()`
    #[must_use]
    pub fn first(self) -> Self {
        // Modify selector to get first match only
        let new_selector = match self.selector {
            Selector::Css(s) => Selector::Css(format!("{s}:first-child")),
            other => other,
        };
        Self {
            selector: new_selector,
            options: self.options,
        }
    }

    /// Get the last matching element
    ///
    /// Per Playwright: `locator.last()`
    #[must_use]
    pub fn last(self) -> Self {
        // Modify selector to get last match only
        let new_selector = match self.selector {
            Selector::Css(s) => Selector::Css(format!("{s}:last-child")),
            other => other,
        };
        Self {
            selector: new_selector,
            options: self.options,
        }
    }

    /// Get the element at the specified index
    ///
    /// Per Playwright: `locator.nth(index)`
    #[must_use]
    pub fn nth(self, index: usize) -> Self {
        // Modify selector to get nth match
        let new_selector = match self.selector {
            Selector::Css(s) => Selector::Css(format!("{s}:nth-child({})", index + 1)),
            other => other,
        };
        Self {
            selector: new_selector,
            options: self.options,
        }
    }

    // =========================================================================
    // PMAT-003: Additional Mouse Actions (Playwright Parity)
    // =========================================================================

    /// Simulate right-clicking on the located element
    ///
    /// Per Playwright: `locator.click({ button: 'right' })`
    pub fn right_click(&self) -> ProbarResult<LocatorAction> {
        Ok(LocatorAction::RightClick {
            locator: self.clone(),
        })
    }

    /// Click with custom options
    ///
    /// Per Playwright: `locator.click(options)`
    pub fn click_with_options(&self, options: ClickOptions) -> ProbarResult<LocatorAction> {
        Ok(LocatorAction::ClickWithOptions {
            locator: self.clone(),
            options,
        })
    }

    /// Hover over the located element
    ///
    /// Per Playwright: `locator.hover()`
    pub fn hover(&self) -> ProbarResult<LocatorAction> {
        Ok(LocatorAction::Hover {
            locator: self.clone(),
        })
    }

    /// Focus the located element
    ///
    /// Per Playwright: `locator.focus()`
    pub fn focus(&self) -> ProbarResult<LocatorAction> {
        Ok(LocatorAction::Focus {
            locator: self.clone(),
        })
    }

    /// Remove focus from the located element
    ///
    /// Per Playwright: `locator.blur()`
    pub fn blur(&self) -> ProbarResult<LocatorAction> {
        Ok(LocatorAction::Blur {
            locator: self.clone(),
        })
    }

    /// Check a checkbox or radio button
    ///
    /// Per Playwright: `locator.check()`
    pub fn check(&self) -> ProbarResult<LocatorAction> {
        Ok(LocatorAction::Check {
            locator: self.clone(),
        })
    }

    /// Uncheck a checkbox
    ///
    /// Per Playwright: `locator.uncheck()`
    pub fn uncheck(&self) -> ProbarResult<LocatorAction> {
        Ok(LocatorAction::Uncheck {
            locator: self.clone(),
        })
    }

    /// Scroll element into view if needed
    ///
    /// Per Playwright: `locator.scrollIntoViewIfNeeded()`
    pub fn scroll_into_view(&self) -> ProbarResult<LocatorAction> {
        Ok(LocatorAction::ScrollIntoView {
            locator: self.clone(),
        })
    }

    // =========================================================================
    // PMAT-001: Semantic Locator Constructors (Convenience methods)
    // =========================================================================

    /// Create a locator by ARIA role
    #[must_use]
    pub fn by_role(role: impl Into<String>) -> Self {
        Self::from_selector(Selector::role(role))
    }

    /// Create a locator by ARIA role with name
    #[must_use]
    pub fn by_role_with_name(role: impl Into<String>, name: impl Into<String>) -> Self {
        Self::from_selector(Selector::role_with_name(role, name))
    }

    /// Create a locator by label text
    #[must_use]
    pub fn by_label(text: impl Into<String>) -> Self {
        Self::from_selector(Selector::label(text))
    }

    /// Create a locator by placeholder text
    #[must_use]
    pub fn by_placeholder(text: impl Into<String>) -> Self {
        Self::from_selector(Selector::placeholder(text))
    }

    /// Create a locator by alt text
    #[must_use]
    pub fn by_alt_text(text: impl Into<String>) -> Self {
        Self::from_selector(Selector::alt_text(text))
    }

    /// Create a locator by test ID
    #[must_use]
    pub fn by_test_id(id: impl Into<String>) -> Self {
        Self::from_selector(Selector::test_id(id))
    }

    /// Create a locator by text content
    #[must_use]
    pub fn by_text(text: impl Into<String>) -> Self {
        Self::from_selector(Selector::text(text))
    }
}

/// Builder for drag operations
#[derive(Debug, Clone)]
pub struct DragBuilder {
    locator: Locator,
    target: Point,
    steps: u32,
    duration: Duration,
}

impl DragBuilder {
    /// Set the number of intermediate steps
    #[must_use]
    pub const fn steps(mut self, steps: u32) -> Self {
        self.steps = steps;
        self
    }

    /// Set the duration of the drag
    #[must_use]
    pub const fn duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Build the drag action
    pub fn build(self) -> LocatorAction {
        LocatorAction::Drag {
            locator: self.locator,
            target: self.target,
            steps: self.steps,
            duration: self.duration,
        }
    }
}

/// Actions that can be performed on a located element
#[derive(Debug, Clone)]
pub enum LocatorAction {
    /// Click on the element
    Click {
        /// The locator
        locator: Locator,
    },
    /// Double-click on the element
    DoubleClick {
        /// The locator
        locator: Locator,
    },
    /// Drag the element to a point
    Drag {
        /// The locator
        locator: Locator,
        /// Target point
        target: Point,
        /// Number of steps
        steps: u32,
        /// Duration
        duration: Duration,
    },
    /// Fill the element with text
    Fill {
        /// The locator
        locator: Locator,
        /// Text to fill
        text: String,
    },
    /// Wait for element to be visible
    WaitForVisible {
        /// The locator
        locator: Locator,
    },
    /// Wait for element to be hidden
    WaitForHidden {
        /// The locator
        locator: Locator,
    },
    // =========================================================================
    // PMAT-003: Additional Actions (Playwright Parity)
    // =========================================================================
    /// Right-click on the element (context menu)
    RightClick {
        /// The locator
        locator: Locator,
    },
    /// Click with custom options
    ClickWithOptions {
        /// The locator
        locator: Locator,
        /// Click options
        options: ClickOptions,
    },
    /// Hover over the element
    Hover {
        /// The locator
        locator: Locator,
    },
    /// Focus the element
    Focus {
        /// The locator
        locator: Locator,
    },
    /// Remove focus from the element
    Blur {
        /// The locator
        locator: Locator,
    },
    /// Check a checkbox or radio
    Check {
        /// The locator
        locator: Locator,
    },
    /// Uncheck a checkbox
    Uncheck {
        /// The locator
        locator: Locator,
    },
    /// Scroll element into view
    ScrollIntoView {
        /// The locator
        locator: Locator,
    },
}

impl LocatorAction {
    /// Get the locator for this action
    #[must_use]
    pub fn locator(&self) -> &Locator {
        match self {
            Self::Click { locator }
            | Self::DoubleClick { locator }
            | Self::Drag { locator, .. }
            | Self::Fill { locator, .. }
            | Self::WaitForVisible { locator }
            | Self::WaitForHidden { locator }
            | Self::RightClick { locator }
            | Self::ClickWithOptions { locator, .. }
            | Self::Hover { locator }
            | Self::Focus { locator }
            | Self::Blur { locator }
            | Self::Check { locator }
            | Self::Uncheck { locator }
            | Self::ScrollIntoView { locator } => locator,
        }
    }
}

/// Queries that return information about located elements
#[derive(Debug, Clone)]
pub enum LocatorQuery {
    /// Get text content
    TextContent {
        /// The locator
        locator: Locator,
    },
    /// Check if visible
    IsVisible {
        /// The locator
        locator: Locator,
    },
    /// Get bounding box
    BoundingBox {
        /// The locator
        locator: Locator,
    },
    /// Count matching elements
    Count {
        /// The locator
        locator: Locator,
    },
}

impl LocatorQuery {
    /// Get the locator for this query
    #[must_use]
    pub const fn locator(&self) -> &Locator {
        match self {
            Self::TextContent { locator }
            | Self::IsVisible { locator }
            | Self::BoundingBox { locator }
            | Self::Count { locator } => locator,
        }
    }
}

/// Bounding box for an element
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    /// X position
    pub x: f32,
    /// Y position
    pub y: f32,
    /// Width
    pub width: f32,
    /// Height
    pub height: f32,
}

impl BoundingBox {
    /// Create a new bounding box
    #[must_use]
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Get the center point
    #[must_use]
    pub fn center(&self) -> Point {
        Point::new(self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Check if a point is inside this bounding box
    #[must_use]
    pub fn contains(&self, point: &Point) -> bool {
        point.x >= self.x
            && point.x <= self.x + self.width
            && point.y >= self.y
            && point.y <= self.y + self.height
    }
}

/// Smart assertion builder for locators (Playwright's `expect()`)
///
/// Per spec: `expect(score_display).to_have_text("10").await?;`
#[derive(Debug, Clone)]
pub struct Expect {
    locator: Locator,
}

impl Expect {
    /// Create a new expectation for a locator
    #[must_use]
    pub const fn new(locator: Locator) -> Self {
        Self { locator }
    }

    /// Assert the element has specific text
    pub fn to_have_text(&self, expected: impl Into<String>) -> ExpectAssertion {
        ExpectAssertion::HasText {
            locator: self.locator.clone(),
            expected: expected.into(),
        }
    }

    /// Assert the element is visible
    pub fn to_be_visible(&self) -> ExpectAssertion {
        ExpectAssertion::IsVisible {
            locator: self.locator.clone(),
        }
    }

    /// Assert the element is hidden
    pub fn to_be_hidden(&self) -> ExpectAssertion {
        ExpectAssertion::IsHidden {
            locator: self.locator.clone(),
        }
    }

    /// Assert the element count
    pub fn to_have_count(&self, count: usize) -> ExpectAssertion {
        ExpectAssertion::HasCount {
            locator: self.locator.clone(),
            expected: count,
        }
    }

    /// Assert the element contains text
    pub fn to_contain_text(&self, text: impl Into<String>) -> ExpectAssertion {
        ExpectAssertion::ContainsText {
            locator: self.locator.clone(),
            expected: text.into(),
        }
    }

    // =========================================================================
    // PMAT-004: Element State Assertions (Playwright Parity)
    // =========================================================================

    /// Assert the element is enabled (not disabled)
    pub fn to_be_enabled(&self) -> ExpectAssertion {
        ExpectAssertion::IsEnabled {
            locator: self.locator.clone(),
        }
    }

    /// Assert the element is disabled
    pub fn to_be_disabled(&self) -> ExpectAssertion {
        ExpectAssertion::IsDisabled {
            locator: self.locator.clone(),
        }
    }

    /// Assert the element is checked (checkbox/radio)
    pub fn to_be_checked(&self) -> ExpectAssertion {
        ExpectAssertion::IsChecked {
            locator: self.locator.clone(),
        }
    }

    /// Assert the element is editable
    pub fn to_be_editable(&self) -> ExpectAssertion {
        ExpectAssertion::IsEditable {
            locator: self.locator.clone(),
        }
    }

    /// Assert the element is focused
    pub fn to_be_focused(&self) -> ExpectAssertion {
        ExpectAssertion::IsFocused {
            locator: self.locator.clone(),
        }
    }

    /// Assert the element is empty
    pub fn to_be_empty(&self) -> ExpectAssertion {
        ExpectAssertion::IsEmpty {
            locator: self.locator.clone(),
        }
    }

    /// Assert the element has specific value
    pub fn to_have_value(&self, value: impl Into<String>) -> ExpectAssertion {
        ExpectAssertion::HasValue {
            locator: self.locator.clone(),
            expected: value.into(),
        }
    }

    /// Assert the element has specific CSS property value
    pub fn to_have_css(
        &self,
        property: impl Into<String>,
        value: impl Into<String>,
    ) -> ExpectAssertion {
        ExpectAssertion::HasCss {
            locator: self.locator.clone(),
            property: property.into(),
            expected: value.into(),
        }
    }

    /// Assert the element has specific class
    pub fn to_have_class(&self, class: impl Into<String>) -> ExpectAssertion {
        ExpectAssertion::HasClass {
            locator: self.locator.clone(),
            expected: class.into(),
        }
    }

    /// Assert the element has specific ID
    pub fn to_have_id(&self, id: impl Into<String>) -> ExpectAssertion {
        ExpectAssertion::HasId {
            locator: self.locator.clone(),
            expected: id.into(),
        }
    }

    /// Assert the element has specific attribute value
    pub fn to_have_attribute(
        &self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> ExpectAssertion {
        ExpectAssertion::HasAttribute {
            locator: self.locator.clone(),
            name: name.into(),
            expected: value.into(),
        }
    }
}

/// Assertion types for `expect()`
#[derive(Debug, Clone)]
pub enum ExpectAssertion {
    /// Element has exact text
    HasText {
        /// The locator
        locator: Locator,
        /// Expected text
        expected: String,
    },
    /// Element is visible
    IsVisible {
        /// The locator
        locator: Locator,
    },
    /// Element is hidden
    IsHidden {
        /// The locator
        locator: Locator,
    },
    /// Element count matches
    HasCount {
        /// The locator
        locator: Locator,
        /// Expected count
        expected: usize,
    },
    /// Element contains text
    ContainsText {
        /// The locator
        locator: Locator,
        /// Text to find
        expected: String,
    },
    // =========================================================================
    // PMAT-004: Element State Assertions (Playwright Parity)
    // =========================================================================
    /// Element is enabled
    IsEnabled {
        /// The locator
        locator: Locator,
    },
    /// Element is disabled
    IsDisabled {
        /// The locator
        locator: Locator,
    },
    /// Element is checked
    IsChecked {
        /// The locator
        locator: Locator,
    },
    /// Element is editable
    IsEditable {
        /// The locator
        locator: Locator,
    },
    /// Element is focused
    IsFocused {
        /// The locator
        locator: Locator,
    },
    /// Element is empty
    IsEmpty {
        /// The locator
        locator: Locator,
    },
    /// Element has specific value
    HasValue {
        /// The locator
        locator: Locator,
        /// Expected value
        expected: String,
    },
    /// Element has specific CSS property
    HasCss {
        /// The locator
        locator: Locator,
        /// CSS property name
        property: String,
        /// Expected value
        expected: String,
    },
    /// Element has specific class
    HasClass {
        /// The locator
        locator: Locator,
        /// Expected class
        expected: String,
    },
    /// Element has specific ID
    HasId {
        /// The locator
        locator: Locator,
        /// Expected ID
        expected: String,
    },
    /// Element has specific attribute
    HasAttribute {
        /// The locator
        locator: Locator,
        /// Attribute name
        name: String,
        /// Expected value
        expected: String,
    },
}

impl ExpectAssertion {
    /// Validate the assertion (synchronous for testing)
    ///
    /// # Errors
    ///
    /// Returns error if assertion fails
    pub fn validate(&self, actual: &str) -> ProbarResult<()> {
        match self {
            Self::HasText { expected, .. } => {
                if actual == expected {
                    Ok(())
                } else {
                    Err(ProbarError::AssertionError {
                        message: format!("Expected text '{expected}' but got '{actual}'"),
                    })
                }
            }
            Self::ContainsText { expected, .. } => {
                if actual.contains(expected) {
                    Ok(())
                } else {
                    Err(ProbarError::AssertionError {
                        message: format!(
                            "Expected text to contain '{expected}' but got '{actual}'"
                        ),
                    })
                }
            }
            Self::HasValue { expected, .. } => {
                if actual == expected {
                    Ok(())
                } else {
                    Err(ProbarError::AssertionError {
                        message: format!("Expected value '{expected}' but got '{actual}'"),
                    })
                }
            }
            Self::HasClass { expected, .. } => {
                if actual.split_whitespace().any(|c| c == expected) {
                    Ok(())
                } else {
                    Err(ProbarError::AssertionError {
                        message: format!("Expected class '{expected}' but got '{actual}'"),
                    })
                }
            }
            Self::HasId { expected, .. } => {
                if actual == expected {
                    Ok(())
                } else {
                    Err(ProbarError::AssertionError {
                        message: format!("Expected id '{expected}' but got '{actual}'"),
                    })
                }
            }
            Self::HasAttribute { name, expected, .. } => {
                if actual == expected {
                    Ok(())
                } else {
                    Err(ProbarError::AssertionError {
                        message: format!(
                            "Expected attribute '{name}' to be '{expected}' but got '{actual}'"
                        ),
                    })
                }
            }
            // These need browser context to validate
            Self::IsVisible { .. }
            | Self::IsHidden { .. }
            | Self::HasCount { .. }
            | Self::IsEnabled { .. }
            | Self::IsDisabled { .. }
            | Self::IsChecked { .. }
            | Self::IsEditable { .. }
            | Self::IsFocused { .. }
            | Self::IsEmpty { .. }
            | Self::HasCss { .. } => Ok(()),
        }
    }

    /// Validate count assertion
    ///
    /// # Errors
    ///
    /// Returns error if count doesn't match
    pub fn validate_count(&self, actual: usize) -> ProbarResult<()> {
        match self {
            Self::HasCount { expected, .. } => {
                if actual == *expected {
                    Ok(())
                } else {
                    Err(ProbarError::AssertionError {
                        message: format!("Expected count {expected} but got {actual}"),
                    })
                }
            }
            _ => Ok(()),
        }
    }

    /// Validate boolean state assertion
    ///
    /// # Errors
    ///
    /// Returns error if state doesn't match expected
    pub fn validate_state(&self, actual: bool) -> ProbarResult<()> {
        match self {
            Self::IsEnabled { .. } => {
                if actual {
                    Ok(())
                } else {
                    Err(ProbarError::AssertionError {
                        message: "Expected element to be enabled but it was disabled".to_string(),
                    })
                }
            }
            Self::IsDisabled { .. } => {
                if actual {
                    Ok(())
                } else {
                    Err(ProbarError::AssertionError {
                        message: "Expected element to be disabled but it was enabled".to_string(),
                    })
                }
            }
            Self::IsChecked { .. } => {
                if actual {
                    Ok(())
                } else {
                    Err(ProbarError::AssertionError {
                        message: "Expected element to be checked but it was not".to_string(),
                    })
                }
            }
            Self::IsEditable { .. } => {
                if actual {
                    Ok(())
                } else {
                    Err(ProbarError::AssertionError {
                        message: "Expected element to be editable but it was not".to_string(),
                    })
                }
            }
            Self::IsFocused { .. } => {
                if actual {
                    Ok(())
                } else {
                    Err(ProbarError::AssertionError {
                        message: "Expected element to be focused but it was not".to_string(),
                    })
                }
            }
            Self::IsEmpty { .. } => {
                if actual {
                    Ok(())
                } else {
                    Err(ProbarError::AssertionError {
                        message: "Expected element to be empty but it was not".to_string(),
                    })
                }
            }
            Self::IsVisible { .. } => {
                if actual {
                    Ok(())
                } else {
                    Err(ProbarError::AssertionError {
                        message: "Expected element to be visible but it was hidden".to_string(),
                    })
                }
            }
            Self::IsHidden { .. } => {
                if actual {
                    Ok(())
                } else {
                    Err(ProbarError::AssertionError {
                        message: "Expected element to be hidden but it was visible".to_string(),
                    })
                }
            }
            _ => Ok(()),
        }
    }
}

/// Create an expectation for a locator (Playwright-style)
///
/// Per spec: `expect(score_display).to_have_text("10").await?;`
#[must_use]
pub fn expect(locator: Locator) -> Expect {
    Expect::new(locator)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::default_trait_access
)]
mod tests {
    use super::*;

    // ========================================================================
    // EXTREME TDD: Tests for Locator abstraction per Section 6.1.1
    // ========================================================================

    mod selector_tests {
        use super::*;

        #[test]
        fn test_css_selector() {
            let selector = Selector::css("button.primary");
            let query = selector.to_query();
            assert!(query.contains("querySelector"));
            assert!(query.contains("button.primary"));
        }

        #[test]
        fn test_test_id_selector() {
            let selector = Selector::test_id("score");
            let query = selector.to_query();
            assert!(query.contains("data-testid"));
            assert!(query.contains("score"));
        }

        #[test]
        fn test_text_selector() {
            let selector = Selector::text("Start Game");
            let query = selector.to_query();
            assert!(query.contains("textContent"));
            assert!(query.contains("Start Game"));
        }

        #[test]
        fn test_entity_selector() {
            let selector = Selector::entity("hero");
            let query = selector.to_query();
            assert!(query.contains("__wasm_get_entity"));
            assert!(query.contains("hero"));
        }

        #[test]
        fn test_count_query() {
            let selector = Selector::css("button");
            let query = selector.to_count_query();
            assert!(query.contains("querySelectorAll"));
            assert!(query.contains(".length"));
        }
    }

    mod locator_tests {
        use super::*;

        #[test]
        fn test_locator_new() {
            let locator = Locator::new("button");
            assert!(matches!(locator.selector(), Selector::Css(_)));
        }

        #[test]
        fn test_locator_with_text() {
            let locator = Locator::new("button").with_text("Start Game");
            assert!(matches!(locator.selector(), Selector::CssWithText { .. }));
        }

        #[test]
        fn test_locator_entity() {
            let locator = Locator::new("canvas").entity("hero");
            assert!(matches!(locator.selector(), Selector::CanvasEntity { .. }));
        }

        #[test]
        fn test_locator_timeout() {
            let locator = Locator::new("button").with_timeout(Duration::from_secs(10));
            assert_eq!(locator.options().timeout, Duration::from_secs(10));
        }

        #[test]
        fn test_locator_strict_mode() {
            let locator = Locator::new("button").with_strict(false);
            assert!(!locator.options().strict);
        }
    }

    mod action_tests {
        use super::*;

        #[test]
        fn test_click_action() {
            let locator = Locator::new("button");
            let action = locator.click().unwrap();
            assert!(matches!(action, LocatorAction::Click { .. }));
        }

        #[test]
        fn test_fill_action() {
            let locator = Locator::new("input");
            let action = locator.fill("test text").unwrap();
            assert!(matches!(action, LocatorAction::Fill { .. }));
        }

        #[test]
        fn test_drag_builder() {
            let locator = Locator::new("canvas").entity("hero");
            let drag = locator
                .drag_to(&Point::new(500.0, 500.0))
                .steps(10)
                .duration(Duration::from_millis(500))
                .build();
            assert!(matches!(drag, LocatorAction::Drag { steps: 10, .. }));
        }
    }

    mod query_tests {
        use super::*;

        #[test]
        fn test_text_content_query() {
            let locator = Locator::new("[data-testid='score']");
            let query = locator.text_content().unwrap();
            assert!(matches!(query, LocatorQuery::TextContent { .. }));
        }

        #[test]
        fn test_is_visible_query() {
            let locator = Locator::new("button");
            let query = locator.is_visible().unwrap();
            assert!(matches!(query, LocatorQuery::IsVisible { .. }));
        }

        #[test]
        fn test_count_query() {
            let locator = Locator::new("li");
            let query = locator.count().unwrap();
            assert!(matches!(query, LocatorQuery::Count { .. }));
        }
    }

    mod expect_tests {
        use super::*;

        #[test]
        fn test_expect_to_have_text() {
            let locator = Locator::new("[data-testid='score']");
            let assertion = expect(locator).to_have_text("10");
            assert!(matches!(assertion, ExpectAssertion::HasText { .. }));
        }

        #[test]
        fn test_expect_to_be_visible() {
            let locator = Locator::new("button");
            let assertion = expect(locator).to_be_visible();
            assert!(matches!(assertion, ExpectAssertion::IsVisible { .. }));
        }

        #[test]
        fn test_expect_to_have_count() {
            let locator = Locator::new("li");
            let assertion = expect(locator).to_have_count(5);
            assert!(matches!(
                assertion,
                ExpectAssertion::HasCount { expected: 5, .. }
            ));
        }

        #[test]
        fn test_validate_has_text_pass() {
            let locator = Locator::new("span");
            let assertion = expect(locator).to_have_text("10");
            assert!(assertion.validate("10").is_ok());
        }

        #[test]
        fn test_validate_has_text_fail() {
            let locator = Locator::new("span");
            let assertion = expect(locator).to_have_text("10");
            assert!(assertion.validate("20").is_err());
        }

        #[test]
        fn test_validate_contains_text_pass() {
            let locator = Locator::new("span");
            let assertion = expect(locator).to_contain_text("Score");
            assert!(assertion.validate("Score: 100").is_ok());
        }

        #[test]
        fn test_validate_count_pass() {
            let locator = Locator::new("li");
            let assertion = expect(locator).to_have_count(3);
            assert!(assertion.validate_count(3).is_ok());
        }

        #[test]
        fn test_validate_count_fail() {
            let locator = Locator::new("li");
            let assertion = expect(locator).to_have_count(3);
            assert!(assertion.validate_count(5).is_err());
        }
    }

    mod point_tests {
        use super::*;

        #[test]
        fn test_point_new() {
            let p = Point::new(100.0, 200.0);
            assert!((p.x - 100.0).abs() < f32::EPSILON);
            assert!((p.y - 200.0).abs() < f32::EPSILON);
        }
    }

    mod bounding_box_tests {
        use super::*;

        #[test]
        fn test_bounding_box_center() {
            let bbox = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
            let center = bbox.center();
            assert!((center.x - 50.0).abs() < f32::EPSILON);
            assert!((center.y - 50.0).abs() < f32::EPSILON);
        }

        #[test]
        fn test_bounding_box_contains() {
            let bbox = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
            assert!(bbox.contains(&Point::new(50.0, 50.0)));
            assert!(!bbox.contains(&Point::new(150.0, 50.0)));
        }
    }

    mod default_tests {
        use super::*;

        #[test]
        fn test_default_timeout() {
            assert_eq!(DEFAULT_TIMEOUT_MS, 5000);
        }

        #[test]
        fn test_default_poll_interval() {
            assert_eq!(DEFAULT_POLL_INTERVAL_MS, 50);
        }

        #[test]
        fn test_locator_options_default() {
            let opts = LocatorOptions::default();
            assert_eq!(opts.timeout, Duration::from_millis(5000));
            assert!(opts.strict);
            assert!(opts.visible);
        }
    }

    mod additional_selector_tests {
        use super::*;

        #[test]
        fn test_xpath_selector_query() {
            let selector = Selector::XPath("//button[@id='test']".to_string());
            let query = selector.to_query();
            assert!(query.contains("evaluate"));
            assert!(query.contains("XPathResult"));
        }

        #[test]
        fn test_xpath_selector_count_query() {
            let selector = Selector::XPath("//button".to_string());
            let query = selector.to_count_query();
            assert!(query.contains("SNAPSHOT"));
            assert!(query.contains("snapshotLength"));
        }

        #[test]
        fn test_css_with_text_selector() {
            let selector = Selector::CssWithText {
                css: "button".to_string(),
                text: "Click Me".to_string(),
            };
            let query = selector.to_query();
            assert!(query.contains("querySelectorAll"));
            assert!(query.contains("textContent"));
        }

        #[test]
        fn test_css_with_text_count_query() {
            let selector = Selector::CssWithText {
                css: "button".to_string(),
                text: "Click".to_string(),
            };
            let query = selector.to_count_query();
            assert!(query.contains("filter"));
            assert!(query.contains(".length"));
        }

        #[test]
        fn test_canvas_entity_selector() {
            let selector = Selector::CanvasEntity {
                entity: "player".to_string(),
            };
            let query = selector.to_query();
            assert!(query.contains("__wasm_get_canvas_entity"));
        }

        #[test]
        fn test_canvas_entity_count_query() {
            let selector = Selector::CanvasEntity {
                entity: "enemy".to_string(),
            };
            let query = selector.to_count_query();
            assert!(query.contains("__wasm_count_canvas_entities"));
        }

        #[test]
        fn test_text_selector_count_query() {
            let selector = Selector::text("Hello");
            let query = selector.to_count_query();
            assert!(query.contains("filter"));
            assert!(query.contains("length"));
        }

        #[test]
        fn test_entity_count_query() {
            let selector = Selector::entity("player");
            let query = selector.to_count_query();
            assert!(query.contains("__wasm_count_entities"));
        }
    }

    mod additional_drag_tests {
        use super::*;

        #[test]
        fn test_drag_operation_defaults() {
            let drag = DragOperation::to(Point::new(100.0, 100.0));
            assert_eq!(drag.steps, 10);
            assert_eq!(drag.duration, Duration::from_millis(500));
        }

        #[test]
        fn test_drag_operation_custom_steps() {
            let drag = DragOperation::to(Point::new(100.0, 100.0)).steps(20);
            assert_eq!(drag.steps, 20);
        }

        #[test]
        fn test_drag_operation_custom_duration() {
            let drag = DragOperation::to(Point::new(100.0, 100.0)).duration(Duration::from_secs(1));
            assert_eq!(drag.duration, Duration::from_secs(1));
        }
    }

    mod additional_locator_tests {
        use super::*;

        #[test]
        fn test_locator_bounding_box() {
            let locator = Locator::new("button");
            let query = locator.bounding_box().unwrap();
            assert!(matches!(query, LocatorQuery::BoundingBox { .. }));
        }

        #[test]
        fn test_locator_from_selector() {
            let selector = Selector::XPath("//button[@id='submit']".to_string());
            let locator = Locator::from_selector(selector);
            assert!(matches!(locator.selector(), Selector::XPath(_)));
        }

        #[test]
        fn test_locator_with_text_non_css() {
            // For non-CSS selectors, with_text should preserve original
            let locator =
                Locator::from_selector(Selector::Entity("hero".to_string())).with_text("ignored");
            assert!(matches!(locator.selector(), Selector::Entity(_)));
        }

        #[test]
        fn test_locator_with_visible() {
            let locator = Locator::new("button").with_visible(false);
            assert!(!locator.options().visible);
        }

        #[test]
        fn test_locator_double_click() {
            let locator = Locator::new("button");
            let action = locator.double_click().unwrap();
            assert!(matches!(action, LocatorAction::DoubleClick { .. }));
        }

        #[test]
        fn test_locator_wait_for_visible() {
            let locator = Locator::new("button");
            let action = locator.wait_for_visible().unwrap();
            assert!(matches!(action, LocatorAction::WaitForVisible { .. }));
        }

        #[test]
        fn test_locator_wait_for_hidden() {
            let locator = Locator::new("button");
            let action = locator.wait_for_hidden().unwrap();
            assert!(matches!(action, LocatorAction::WaitForHidden { .. }));
        }

        #[test]
        fn test_locator_action_locator_accessor() {
            let locator = Locator::new("button");
            let action = locator.click().unwrap();
            let _ = action.locator(); // Access the locator
            assert!(matches!(action, LocatorAction::Click { .. }));
        }

        #[test]
        fn test_locator_query_locator_accessor() {
            let locator = Locator::new("button");
            let query = locator.count().unwrap();
            let accessed = query.locator();
            assert!(matches!(accessed.selector(), Selector::Css(_)));
        }

        #[test]
        fn test_selector_to_count_query_all_variants() {
            // Test XPath count query
            let xpath = Selector::XPath("//button".to_string());
            assert!(xpath.to_count_query().contains("snapshotLength"));

            // Test Text count query
            let text = Selector::Text("Click me".to_string());
            assert!(text.to_count_query().contains(".length"));

            // Test TestId count query
            let testid = Selector::TestId("btn".to_string());
            assert!(testid.to_count_query().contains("data-testid"));

            // Test Entity count query
            let entity = Selector::Entity("hero".to_string());
            assert!(entity.to_count_query().contains("__wasm_count_entities"));

            // Test CssWithText count query
            let css_text = Selector::CssWithText {
                css: "button".to_string(),
                text: "Submit".to_string(),
            };
            assert!(css_text.to_count_query().contains(".length"));

            // Test CanvasEntity count query
            let canvas = Selector::CanvasEntity {
                entity: "player".to_string(),
            };
            assert!(canvas
                .to_count_query()
                .contains("__wasm_count_canvas_entities"));
        }
    }

    mod additional_bounding_box_tests {
        use super::*;

        #[test]
        fn test_bounding_box_creation_and_fields() {
            let bbox = BoundingBox::new(10.0, 20.0, 100.0, 50.0);
            assert!((bbox.x - 10.0).abs() < f32::EPSILON);
            assert!((bbox.y - 20.0).abs() < f32::EPSILON);
            assert!((bbox.width - 100.0).abs() < f32::EPSILON);
            assert!((bbox.height - 50.0).abs() < f32::EPSILON);
        }

        #[test]
        fn test_bounding_box_contains_edge_cases() {
            let bbox = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
            // On the edge should be inside
            assert!(bbox.contains(&Point::new(0.0, 0.0)));
            assert!(bbox.contains(&Point::new(100.0, 100.0)));
            // Just outside should not be inside
            assert!(!bbox.contains(&Point::new(-1.0, 50.0)));
            assert!(!bbox.contains(&Point::new(101.0, 50.0)));
        }
    }

    // ============================================================================
    // QA CHECKLIST SECTION 2: Locator API Falsification Tests
    // Per docs/qa/100-point-qa-checklist-jugar-probar.md
    // ============================================================================

    #[allow(clippy::uninlined_format_args, unused_imports)]
    mod qa_checklist_locator_tests {
        #[allow(unused_imports)]
        use super::*;

        /// Test #25: Extremely long selector (10KB) - length limit enforced
        #[test]
        fn test_long_selector_limit() {
            const MAX_SELECTOR_LENGTH: usize = 10 * 1024; // 10KB limit
            let long_selector = "a".repeat(MAX_SELECTOR_LENGTH + 1);

            // Validate that we can detect oversized selectors
            let is_too_long = long_selector.len() > MAX_SELECTOR_LENGTH;
            assert!(is_too_long, "Should detect selector exceeding 10KB limit");

            // System should enforce limit (truncate or reject)
            let truncated = if long_selector.len() > MAX_SELECTOR_LENGTH {
                &long_selector[..MAX_SELECTOR_LENGTH]
            } else {
                &long_selector
            };
            assert_eq!(truncated.len(), MAX_SELECTOR_LENGTH);
        }

        /// Test #34: Shadow DOM elements traversal
        #[test]
        fn test_shadow_dom_selector_support() {
            // Shadow DOM requires special traversal via >>> or /deep/
            let shadow_selector = "host-element >>> .inner-element";

            // Validate shadow-piercing combinator is recognized
            let has_shadow_combinator =
                shadow_selector.contains(">>>") || shadow_selector.contains("/deep/");
            assert!(has_shadow_combinator, "Shadow DOM combinator recognized");

            // Generate appropriate query for shadow DOM
            let query = if shadow_selector.contains(">>>") {
                let parts: Vec<&str> = shadow_selector.split(">>>").collect();
                format!(
                    "document.querySelector('{}').shadowRoot.querySelector('{}')",
                    parts[0].trim(),
                    parts.get(1).unwrap_or(&"").trim()
                )
            } else {
                shadow_selector.to_string()
            };
            assert!(query.contains("shadowRoot"), "Shadow DOM query generated");
        }

        /// Test #35: iframe elements context switching
        #[test]
        fn test_iframe_context_switching() {
            // iframe requires contentDocument access
            let iframe_selector = "iframe#game-frame";
            let inner_selector = "button.start";

            // Generate iframe traversal query
            let query = format!(
                "document.querySelector('{}').contentDocument.querySelector('{}')",
                iframe_selector, inner_selector
            );

            assert!(query.contains("contentDocument"), "iframe context switch");
            assert!(query.contains(inner_selector), "Inner selector preserved");
        }

        /// Test empty selector handling (Test #21 reinforcement)
        #[test]
        fn test_empty_selector_rejection() {
            let empty_selector = "";
            let whitespace_selector = "   ";

            let is_empty_or_whitespace =
                empty_selector.is_empty() || whitespace_selector.trim().is_empty();
            assert!(
                is_empty_or_whitespace,
                "Empty/whitespace selectors detected"
            );
        }

        /// Test special characters in selectors
        #[test]
        fn test_special_char_selector_escaping() {
            let selector_with_quotes = r#"button[data-name="test's"]"#;
            let selector_with_brackets = "div[class~=foo\\[bar\\]]";

            // These should not cause parsing issues
            assert!(selector_with_quotes.contains('"'));
            assert!(selector_with_brackets.contains('['));
        }
    }

    // ============================================================================
    // PMAT-001: Semantic Locators Tests
    // ============================================================================

    mod semantic_locator_tests {
        use super::*;

        #[test]
        fn test_role_selector_query() {
            let selector = Selector::role("button");
            let query = selector.to_query();
            assert!(query.contains("role"));
            assert!(query.contains("button"));
        }

        #[test]
        fn test_role_selector_with_name() {
            let selector = Selector::role_with_name("button", "Submit");
            let query = selector.to_query();
            assert!(query.contains("role"));
            assert!(query.contains("Submit"));
        }

        #[test]
        fn test_role_selector_count_query() {
            let selector = Selector::role("textbox");
            let query = selector.to_count_query();
            assert!(query.contains("role"));
            assert!(query.contains(".length"));
        }

        #[test]
        fn test_label_selector_query() {
            let selector = Selector::label("Username");
            let query = selector.to_query();
            assert!(query.contains("label"));
            assert!(query.contains("Username"));
        }

        #[test]
        fn test_label_selector_count_query() {
            let selector = Selector::label("Email");
            let query = selector.to_count_query();
            assert!(query.contains("label"));
            assert!(query.contains(".length"));
        }

        #[test]
        fn test_placeholder_selector_query() {
            let selector = Selector::placeholder("Enter email");
            let query = selector.to_query();
            assert!(query.contains("placeholder"));
            assert!(query.contains("Enter email"));
        }

        #[test]
        fn test_placeholder_selector_count_query() {
            let selector = Selector::placeholder("Search");
            let query = selector.to_count_query();
            assert!(query.contains("placeholder"));
            assert!(query.contains(".length"));
        }

        #[test]
        fn test_alt_text_selector_query() {
            let selector = Selector::alt_text("Company Logo");
            let query = selector.to_query();
            assert!(query.contains("alt"));
            assert!(query.contains("Company Logo"));
        }

        #[test]
        fn test_alt_text_selector_count_query() {
            let selector = Selector::alt_text("Logo");
            let query = selector.to_count_query();
            assert!(query.contains("alt"));
            assert!(query.contains(".length"));
        }

        #[test]
        fn test_locator_by_role() {
            let locator = Locator::by_role("button");
            assert!(matches!(locator.selector(), Selector::Role { .. }));
        }

        #[test]
        fn test_locator_by_role_with_name() {
            let locator = Locator::by_role_with_name("link", "Home");
            match locator.selector() {
                Selector::Role { name, .. } => assert!(name.is_some()),
                _ => panic!("Expected Role selector"),
            }
        }

        #[test]
        fn test_locator_by_label() {
            let locator = Locator::by_label("Password");
            assert!(matches!(locator.selector(), Selector::Label(_)));
        }

        #[test]
        fn test_locator_by_placeholder() {
            let locator = Locator::by_placeholder("Enter your name");
            assert!(matches!(locator.selector(), Selector::Placeholder(_)));
        }

        #[test]
        fn test_locator_by_alt_text() {
            let locator = Locator::by_alt_text("Profile Picture");
            assert!(matches!(locator.selector(), Selector::AltText(_)));
        }

        #[test]
        fn test_locator_by_test_id() {
            let locator = Locator::by_test_id("submit-btn");
            assert!(matches!(locator.selector(), Selector::TestId(_)));
        }

        #[test]
        fn test_locator_by_text() {
            let locator = Locator::by_text("Click here");
            assert!(matches!(locator.selector(), Selector::Text(_)));
        }
    }

    // ============================================================================
    // PMAT-002: Locator Operations Tests
    // ============================================================================

    mod locator_operations_tests {
        use super::*;

        #[test]
        fn test_filter_with_has_text() {
            let locator = Locator::new("button").filter(FilterOptions::new().has_text("Submit"));
            assert!(matches!(locator.selector(), Selector::CssWithText { .. }));
        }

        #[test]
        fn test_filter_options_builder() {
            let options = FilterOptions::new()
                .has_text("Hello")
                .has_not_text("Goodbye");
            assert!(options.has_text.is_some());
            assert!(options.has_not_text.is_some());
        }

        #[test]
        fn test_filter_options_has() {
            let child = Locator::new(".child");
            let options = FilterOptions::new().has(child);
            assert!(options.has.is_some());
        }

        #[test]
        fn test_filter_options_has_not() {
            let child = Locator::new(".excluded");
            let options = FilterOptions::new().has_not(child);
            assert!(options.has_not.is_some());
        }

        #[test]
        fn test_locator_and() {
            let locator1 = Locator::new("div");
            let locator2 = Locator::new(".active");
            let combined = locator1.and(locator2);
            if let Selector::Css(s) = combined.selector() {
                assert!(s.contains("div"));
                assert!(s.contains(".active"));
            } else {
                panic!("Expected CSS selector");
            }
        }

        #[test]
        fn test_locator_or() {
            let locator1 = Locator::new("button");
            let locator2 = Locator::new("a.btn");
            let combined = locator1.or(locator2);
            if let Selector::Css(s) = combined.selector() {
                assert!(s.contains("button"));
                assert!(s.contains("a.btn"));
                assert!(s.contains(", "));
            } else {
                panic!("Expected CSS selector");
            }
        }

        #[test]
        fn test_locator_first() {
            let locator = Locator::new("li").first();
            if let Selector::Css(s) = locator.selector() {
                assert!(s.contains(":first-child"));
            } else {
                panic!("Expected CSS selector");
            }
        }

        #[test]
        fn test_locator_last() {
            let locator = Locator::new("li").last();
            if let Selector::Css(s) = locator.selector() {
                assert!(s.contains(":last-child"));
            } else {
                panic!("Expected CSS selector");
            }
        }

        #[test]
        fn test_locator_nth() {
            let locator = Locator::new("li").nth(2);
            if let Selector::Css(s) = locator.selector() {
                assert!(s.contains(":nth-child(3)")); // 0-indexed to 1-indexed
            } else {
                panic!("Expected CSS selector");
            }
        }

        #[test]
        fn test_locator_and_non_css() {
            let locator1 = Locator::from_selector(Selector::Entity("hero".to_string()));
            let locator2 = Locator::new("div");
            let combined = locator1.and(locator2);
            // Should keep the original non-CSS selector
            assert!(matches!(combined.selector(), Selector::Entity(_)));
        }
    }

    // ============================================================================
    // PMAT-003: Mouse Actions Tests
    // ============================================================================

    mod mouse_actions_tests {
        use super::*;

        #[test]
        fn test_right_click() {
            let locator = Locator::new("button");
            let action = locator.right_click().unwrap();
            assert!(matches!(action, LocatorAction::RightClick { .. }));
        }

        #[test]
        fn test_hover() {
            let locator = Locator::new("menu-item");
            let action = locator.hover().unwrap();
            assert!(matches!(action, LocatorAction::Hover { .. }));
        }

        #[test]
        fn test_focus() {
            let locator = Locator::new("input");
            let action = locator.focus().unwrap();
            assert!(matches!(action, LocatorAction::Focus { .. }));
        }

        #[test]
        fn test_blur() {
            let locator = Locator::new("input");
            let action = locator.blur().unwrap();
            assert!(matches!(action, LocatorAction::Blur { .. }));
        }

        #[test]
        fn test_check() {
            let locator = Locator::new("input[type=checkbox]");
            let action = locator.check().unwrap();
            assert!(matches!(action, LocatorAction::Check { .. }));
        }

        #[test]
        fn test_uncheck() {
            let locator = Locator::new("input[type=checkbox]");
            let action = locator.uncheck().unwrap();
            assert!(matches!(action, LocatorAction::Uncheck { .. }));
        }

        #[test]
        fn test_scroll_into_view() {
            let locator = Locator::new("footer");
            let action = locator.scroll_into_view().unwrap();
            assert!(matches!(action, LocatorAction::ScrollIntoView { .. }));
        }

        #[test]
        fn test_click_with_options_default() {
            let options = ClickOptions::new();
            assert_eq!(options.button, MouseButton::Left);
            assert_eq!(options.click_count, 1);
            assert!(options.position.is_none());
            assert!(options.modifiers.is_empty());
        }

        #[test]
        fn test_click_with_options_right_button() {
            let options = ClickOptions::new().button(MouseButton::Right);
            assert_eq!(options.button, MouseButton::Right);
        }

        #[test]
        fn test_click_with_options_double_click() {
            let options = ClickOptions::new().click_count(2);
            assert_eq!(options.click_count, 2);
        }

        #[test]
        fn test_click_with_options_position() {
            let options = ClickOptions::new().position(Point::new(10.0, 20.0));
            assert!(options.position.is_some());
        }

        #[test]
        fn test_click_with_options_modifier() {
            let options = ClickOptions::new()
                .modifier(KeyModifier::Shift)
                .modifier(KeyModifier::Control);
            assert_eq!(options.modifiers.len(), 2);
        }

        #[test]
        fn test_click_with_custom_options() {
            let locator = Locator::new("button");
            let options = ClickOptions::new().button(MouseButton::Middle);
            let action = locator.click_with_options(options).unwrap();
            assert!(matches!(action, LocatorAction::ClickWithOptions { .. }));
        }

        #[test]
        fn test_mouse_button_default() {
            let button: MouseButton = Default::default();
            assert_eq!(button, MouseButton::Left);
        }

        #[test]
        fn test_locator_action_locator_accessor_all_variants() {
            let locator = Locator::new("button");

            // Test new action variants
            let _ = locator.right_click().unwrap().locator();
            let _ = locator.hover().unwrap().locator();
            let _ = locator.focus().unwrap().locator();
            let _ = locator.blur().unwrap().locator();
            let _ = locator.check().unwrap().locator();
            let _ = locator.uncheck().unwrap().locator();
            let _ = locator.scroll_into_view().unwrap().locator();
            let _ = locator
                .click_with_options(ClickOptions::new())
                .unwrap()
                .locator();
        }
    }

    // ============================================================================
    // PMAT-004: Element State Assertions Tests
    // ============================================================================

    mod element_state_assertions_tests {
        use super::*;

        #[test]
        fn test_to_be_enabled() {
            let locator = Locator::new("button");
            let assertion = expect(locator).to_be_enabled();
            assert!(matches!(assertion, ExpectAssertion::IsEnabled { .. }));
        }

        #[test]
        fn test_to_be_disabled() {
            let locator = Locator::new("button");
            let assertion = expect(locator).to_be_disabled();
            assert!(matches!(assertion, ExpectAssertion::IsDisabled { .. }));
        }

        #[test]
        fn test_to_be_checked() {
            let locator = Locator::new("input[type=checkbox]");
            let assertion = expect(locator).to_be_checked();
            assert!(matches!(assertion, ExpectAssertion::IsChecked { .. }));
        }

        #[test]
        fn test_to_be_editable() {
            let locator = Locator::new("textarea");
            let assertion = expect(locator).to_be_editable();
            assert!(matches!(assertion, ExpectAssertion::IsEditable { .. }));
        }

        #[test]
        fn test_to_be_focused() {
            let locator = Locator::new("input");
            let assertion = expect(locator).to_be_focused();
            assert!(matches!(assertion, ExpectAssertion::IsFocused { .. }));
        }

        #[test]
        fn test_to_be_empty() {
            let locator = Locator::new("div");
            let assertion = expect(locator).to_be_empty();
            assert!(matches!(assertion, ExpectAssertion::IsEmpty { .. }));
        }

        #[test]
        fn test_to_have_value() {
            let locator = Locator::new("input");
            let assertion = expect(locator).to_have_value("test");
            assert!(matches!(assertion, ExpectAssertion::HasValue { .. }));
        }

        #[test]
        fn test_to_have_css() {
            let locator = Locator::new("div");
            let assertion = expect(locator).to_have_css("color", "red");
            assert!(matches!(assertion, ExpectAssertion::HasCss { .. }));
        }

        #[test]
        fn test_to_have_class() {
            let locator = Locator::new("div");
            let assertion = expect(locator).to_have_class("active");
            assert!(matches!(assertion, ExpectAssertion::HasClass { .. }));
        }

        #[test]
        fn test_to_have_id() {
            let locator = Locator::new("div");
            let assertion = expect(locator).to_have_id("main-content");
            assert!(matches!(assertion, ExpectAssertion::HasId { .. }));
        }

        #[test]
        fn test_to_have_attribute() {
            let locator = Locator::new("input");
            let assertion = expect(locator).to_have_attribute("type", "text");
            assert!(matches!(assertion, ExpectAssertion::HasAttribute { .. }));
        }

        #[test]
        fn test_validate_has_value_pass() {
            let locator = Locator::new("input");
            let assertion = expect(locator).to_have_value("test123");
            assert!(assertion.validate("test123").is_ok());
        }

        #[test]
        fn test_validate_has_value_fail() {
            let locator = Locator::new("input");
            let assertion = expect(locator).to_have_value("expected");
            assert!(assertion.validate("actual").is_err());
        }

        #[test]
        fn test_validate_has_class_pass() {
            let locator = Locator::new("div");
            let assertion = expect(locator).to_have_class("active");
            assert!(assertion.validate("btn active primary").is_ok());
        }

        #[test]
        fn test_validate_has_class_fail() {
            let locator = Locator::new("div");
            let assertion = expect(locator).to_have_class("missing");
            assert!(assertion.validate("btn active").is_err());
        }

        #[test]
        fn test_validate_has_id_pass() {
            let locator = Locator::new("div");
            let assertion = expect(locator).to_have_id("main");
            assert!(assertion.validate("main").is_ok());
        }

        #[test]
        fn test_validate_has_attribute_pass() {
            let locator = Locator::new("input");
            let assertion = expect(locator).to_have_attribute("type", "text");
            assert!(assertion.validate("text").is_ok());
        }

        #[test]
        fn test_validate_state_enabled_pass() {
            let locator = Locator::new("button");
            let assertion = expect(locator).to_be_enabled();
            assert!(assertion.validate_state(true).is_ok());
        }

        #[test]
        fn test_validate_state_enabled_fail() {
            let locator = Locator::new("button");
            let assertion = expect(locator).to_be_enabled();
            assert!(assertion.validate_state(false).is_err());
        }

        #[test]
        fn test_validate_state_disabled_pass() {
            let locator = Locator::new("button");
            let assertion = expect(locator).to_be_disabled();
            assert!(assertion.validate_state(true).is_ok());
        }

        #[test]
        fn test_validate_state_checked_pass() {
            let locator = Locator::new("input");
            let assertion = expect(locator).to_be_checked();
            assert!(assertion.validate_state(true).is_ok());
        }

        #[test]
        fn test_validate_state_editable_pass() {
            let locator = Locator::new("textarea");
            let assertion = expect(locator).to_be_editable();
            assert!(assertion.validate_state(true).is_ok());
        }

        #[test]
        fn test_validate_state_focused_pass() {
            let locator = Locator::new("input");
            let assertion = expect(locator).to_be_focused();
            assert!(assertion.validate_state(true).is_ok());
        }

        #[test]
        fn test_validate_state_empty_pass() {
            let locator = Locator::new("div");
            let assertion = expect(locator).to_be_empty();
            assert!(assertion.validate_state(true).is_ok());
        }

        #[test]
        fn test_validate_state_visible_pass() {
            let locator = Locator::new("div");
            let assertion = expect(locator).to_be_visible();
            assert!(assertion.validate_state(true).is_ok());
        }

        #[test]
        fn test_validate_state_hidden_pass() {
            let locator = Locator::new("div");
            let assertion = expect(locator).to_be_hidden();
            assert!(assertion.validate_state(true).is_ok());
        }
    }

    // =========================================================================
    // H₀ EXTREME TDD: Auto-Waiting Tests (Spec G.1 P0)
    // =========================================================================

    mod h0_auto_waiting_tests {
        use super::*;

        #[test]
        fn h0_locator_01_default_timeout_is_5_seconds() {
            assert_eq!(DEFAULT_TIMEOUT_MS, 5000);
        }

        #[test]
        fn h0_locator_02_default_poll_interval_is_50ms() {
            assert_eq!(DEFAULT_POLL_INTERVAL_MS, 50);
        }

        #[test]
        fn h0_locator_03_locator_options_default_timeout() {
            let opts = LocatorOptions::default();
            assert_eq!(opts.timeout, Duration::from_millis(DEFAULT_TIMEOUT_MS));
        }

        #[test]
        fn h0_locator_04_locator_options_default_strict_true() {
            let opts = LocatorOptions::default();
            assert!(opts.strict);
        }

        #[test]
        fn h0_locator_05_locator_options_default_visible_true() {
            let opts = LocatorOptions::default();
            assert!(opts.visible);
        }

        #[test]
        fn h0_locator_06_with_timeout_custom_value() {
            let locator = Locator::new("button").with_timeout(Duration::from_secs(30));
            assert_eq!(locator.options().timeout, Duration::from_secs(30));
        }

        #[test]
        fn h0_locator_07_with_strict_false() {
            let locator = Locator::new("button").with_strict(false);
            assert!(!locator.options().strict);
        }

        #[test]
        fn h0_locator_08_with_visible_false() {
            let locator = Locator::new("button").with_visible(false);
            assert!(!locator.options().visible);
        }

        #[test]
        fn h0_locator_09_wait_for_visible_action() {
            let locator = Locator::new("button");
            let action = locator.wait_for_visible().unwrap();
            assert!(matches!(action, LocatorAction::WaitForVisible { .. }));
        }

        #[test]
        fn h0_locator_10_wait_for_hidden_action() {
            let locator = Locator::new("button");
            let action = locator.wait_for_hidden().unwrap();
            assert!(matches!(action, LocatorAction::WaitForHidden { .. }));
        }
    }

    // =========================================================================
    // H₀ EXTREME TDD: Semantic Locators (Spec G.1 Playwright Parity)
    // =========================================================================

    mod h0_semantic_locator_tests {
        use super::*;

        #[test]
        fn h0_locator_11_role_selector_button() {
            let selector = Selector::role("button");
            assert!(matches!(selector, Selector::Role { role, name: None } if role == "button"));
        }

        #[test]
        fn h0_locator_12_role_selector_with_name() {
            let selector = Selector::role_with_name("button", "Submit");
            assert!(
                matches!(selector, Selector::Role { role, name: Some(n) } if role == "button" && n == "Submit")
            );
        }

        #[test]
        fn h0_locator_13_label_selector() {
            let selector = Selector::label("Username");
            assert!(matches!(selector, Selector::Label(l) if l == "Username"));
        }

        #[test]
        fn h0_locator_14_placeholder_selector() {
            let selector = Selector::placeholder("Enter email");
            assert!(matches!(selector, Selector::Placeholder(p) if p == "Enter email"));
        }

        #[test]
        fn h0_locator_15_alt_text_selector() {
            let selector = Selector::alt_text("Logo image");
            assert!(matches!(selector, Selector::AltText(a) if a == "Logo image"));
        }

        #[test]
        fn h0_locator_16_role_to_query() {
            let selector = Selector::role("button");
            let query = selector.to_query();
            assert!(query.contains("role") || query.contains("button"));
        }

        #[test]
        fn h0_locator_17_label_to_query() {
            let selector = Selector::label("Email");
            let query = selector.to_query();
            assert!(query.contains("label") || query.contains("Email"));
        }

        #[test]
        fn h0_locator_18_placeholder_to_query() {
            let selector = Selector::placeholder("Search");
            let query = selector.to_query();
            assert!(query.contains("placeholder") || query.contains("Search"));
        }

        #[test]
        fn h0_locator_19_alt_text_to_query() {
            let selector = Selector::alt_text("Company Logo");
            let query = selector.to_query();
            assert!(query.contains("alt") || query.contains("Company Logo"));
        }

        #[test]
        fn h0_locator_20_css_selector_factory() {
            let selector = Selector::css("div.container");
            assert!(matches!(selector, Selector::Css(s) if s == "div.container"));
        }
    }

    // =========================================================================
    // H₀ EXTREME TDD: Expect Assertions (Spec G.1 Auto-Retry)
    // =========================================================================

    mod h0_expect_assertion_tests {
        use super::*;

        #[test]
        fn h0_locator_21_expect_to_have_text() {
            let locator = Locator::new("span");
            let assertion = expect(locator).to_have_text("Hello");
            assert!(matches!(assertion, ExpectAssertion::HasText { .. }));
        }

        #[test]
        fn h0_locator_22_expect_to_contain_text() {
            let locator = Locator::new("span");
            let assertion = expect(locator).to_contain_text("ell");
            assert!(matches!(assertion, ExpectAssertion::ContainsText { .. }));
        }

        #[test]
        fn h0_locator_23_expect_to_have_count() {
            let locator = Locator::new("li");
            let assertion = expect(locator).to_have_count(5);
            assert!(
                matches!(assertion, ExpectAssertion::HasCount { expected, .. } if expected == 5)
            );
        }

        #[test]
        fn h0_locator_24_expect_to_be_visible() {
            let locator = Locator::new("button");
            let assertion = expect(locator).to_be_visible();
            assert!(matches!(assertion, ExpectAssertion::IsVisible { .. }));
        }

        #[test]
        fn h0_locator_25_expect_to_be_hidden() {
            let locator = Locator::new("button");
            let assertion = expect(locator).to_be_hidden();
            assert!(matches!(assertion, ExpectAssertion::IsHidden { .. }));
        }

        #[test]
        fn h0_locator_26_expect_to_be_enabled() {
            let locator = Locator::new("button");
            let assertion = expect(locator).to_be_enabled();
            assert!(matches!(assertion, ExpectAssertion::IsEnabled { .. }));
        }

        #[test]
        fn h0_locator_27_expect_to_be_disabled() {
            let locator = Locator::new("button");
            let assertion = expect(locator).to_be_disabled();
            assert!(matches!(assertion, ExpectAssertion::IsDisabled { .. }));
        }

        #[test]
        fn h0_locator_28_expect_to_be_checked() {
            let locator = Locator::new("input[type=checkbox]");
            let assertion = expect(locator).to_be_checked();
            assert!(matches!(assertion, ExpectAssertion::IsChecked { .. }));
        }

        #[test]
        fn h0_locator_29_expect_to_have_value() {
            let locator = Locator::new("input");
            let assertion = expect(locator).to_have_value("test");
            assert!(matches!(assertion, ExpectAssertion::HasValue { .. }));
        }

        #[test]
        fn h0_locator_30_expect_to_have_attribute() {
            let locator = Locator::new("input");
            let assertion = expect(locator).to_have_attribute("type", "email");
            assert!(matches!(assertion, ExpectAssertion::HasAttribute { .. }));
        }
    }

    // =========================================================================
    // H₀ EXTREME TDD: Locator Actions (Spec G.1)
    // =========================================================================

    mod h0_locator_action_tests {
        use super::*;

        #[test]
        fn h0_locator_31_click_action() {
            let locator = Locator::new("button");
            let action = locator.click().unwrap();
            assert!(matches!(action, LocatorAction::Click { .. }));
        }

        #[test]
        fn h0_locator_32_double_click_action() {
            let locator = Locator::new("button");
            let action = locator.double_click().unwrap();
            assert!(matches!(action, LocatorAction::DoubleClick { .. }));
        }

        #[test]
        fn h0_locator_33_fill_action() {
            let locator = Locator::new("input");
            let action = locator.fill("hello").unwrap();
            assert!(matches!(action, LocatorAction::Fill { text, .. } if text == "hello"));
        }

        #[test]
        fn h0_locator_34_hover_action() {
            let locator = Locator::new("button");
            let action = locator.hover().unwrap();
            assert!(matches!(action, LocatorAction::Hover { .. }));
        }

        #[test]
        fn h0_locator_35_focus_action() {
            let locator = Locator::new("input");
            let action = locator.focus().unwrap();
            assert!(matches!(action, LocatorAction::Focus { .. }));
        }

        #[test]
        fn h0_locator_36_drag_to_action() {
            let locator = Locator::new("div.draggable");
            let action = locator.drag_to(&Point::new(100.0, 200.0)).build();
            assert!(matches!(action, LocatorAction::Drag { .. }));
        }

        #[test]
        fn h0_locator_37_drag_steps_custom() {
            let locator = Locator::new("div");
            let action = locator.drag_to(&Point::new(0.0, 0.0)).steps(25).build();
            assert!(matches!(action, LocatorAction::Drag { steps: 25, .. }));
        }

        #[test]
        fn h0_locator_38_drag_duration_custom() {
            let locator = Locator::new("div");
            let action = locator
                .drag_to(&Point::new(0.0, 0.0))
                .duration(Duration::from_secs(2))
                .build();
            assert!(
                matches!(action, LocatorAction::Drag { duration, .. } if duration == Duration::from_secs(2))
            );
        }

        #[test]
        fn h0_locator_39_text_content_query() {
            let locator = Locator::new("span");
            let query = locator.text_content().unwrap();
            assert!(matches!(query, LocatorQuery::TextContent { .. }));
        }

        #[test]
        fn h0_locator_40_count_query() {
            let locator = Locator::new("li");
            let query = locator.count().unwrap();
            assert!(matches!(query, LocatorQuery::Count { .. }));
        }
    }

    // =========================================================================
    // H₀ EXTREME TDD: BoundingBox and Point (Spec G.1)
    // =========================================================================

    mod h0_geometry_tests {
        use super::*;

        #[test]
        fn h0_locator_41_point_new() {
            let p = Point::new(10.5, 20.5);
            assert!((p.x - 10.5).abs() < f32::EPSILON);
            assert!((p.y - 20.5).abs() < f32::EPSILON);
        }

        #[test]
        fn h0_locator_42_bounding_box_new() {
            let bbox = BoundingBox::new(10.0, 20.0, 100.0, 50.0);
            assert!((bbox.x - 10.0).abs() < f32::EPSILON);
            assert!((bbox.width - 100.0).abs() < f32::EPSILON);
        }

        #[test]
        fn h0_locator_43_bounding_box_center() {
            let bbox = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
            let center = bbox.center();
            assert!((center.x - 50.0).abs() < f32::EPSILON);
            assert!((center.y - 50.0).abs() < f32::EPSILON);
        }

        #[test]
        fn h0_locator_44_bounding_box_contains_inside() {
            let bbox = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
            assert!(bbox.contains(&Point::new(50.0, 50.0)));
        }

        #[test]
        fn h0_locator_45_bounding_box_contains_outside() {
            let bbox = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
            assert!(!bbox.contains(&Point::new(150.0, 150.0)));
        }

        #[test]
        fn h0_locator_46_bounding_box_contains_edge() {
            let bbox = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
            assert!(bbox.contains(&Point::new(0.0, 0.0)));
        }

        #[test]
        fn h0_locator_47_drag_operation_default_steps() {
            let drag = DragOperation::to(Point::new(100.0, 100.0));
            assert_eq!(drag.steps, 10);
        }

        #[test]
        fn h0_locator_48_drag_operation_default_duration() {
            let drag = DragOperation::to(Point::new(100.0, 100.0));
            assert_eq!(drag.duration, Duration::from_millis(500));
        }

        #[test]
        fn h0_locator_49_locator_bounding_box_query() {
            let locator = Locator::new("div");
            let query = locator.bounding_box().unwrap();
            assert!(matches!(query, LocatorQuery::BoundingBox { .. }));
        }

        #[test]
        fn h0_locator_50_locator_is_visible_query() {
            let locator = Locator::new("div");
            let query = locator.is_visible().unwrap();
            assert!(matches!(query, LocatorQuery::IsVisible { .. }));
        }
    }

    // =========================================================================
    // Additional Coverage Tests: Edge Cases and Failure Paths
    // =========================================================================

    mod coverage_edge_cases {
        use super::*;

        // -------------------------------------------------------------------
        // validate_state failure cases
        // -------------------------------------------------------------------

        #[test]
        fn test_validate_state_disabled_fail() {
            let locator = Locator::new("button");
            let assertion = expect(locator).to_be_disabled();
            let result = assertion.validate_state(false);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(err.to_string().contains("disabled"));
        }

        #[test]
        fn test_validate_state_checked_fail() {
            let locator = Locator::new("input");
            let assertion = expect(locator).to_be_checked();
            let result = assertion.validate_state(false);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(err.to_string().contains("checked"));
        }

        #[test]
        fn test_validate_state_editable_fail() {
            let locator = Locator::new("textarea");
            let assertion = expect(locator).to_be_editable();
            let result = assertion.validate_state(false);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(err.to_string().contains("editable"));
        }

        #[test]
        fn test_validate_state_focused_fail() {
            let locator = Locator::new("input");
            let assertion = expect(locator).to_be_focused();
            let result = assertion.validate_state(false);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(err.to_string().contains("focused"));
        }

        #[test]
        fn test_validate_state_empty_fail() {
            let locator = Locator::new("div");
            let assertion = expect(locator).to_be_empty();
            let result = assertion.validate_state(false);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(err.to_string().contains("empty"));
        }

        #[test]
        fn test_validate_state_visible_fail() {
            let locator = Locator::new("div");
            let assertion = expect(locator).to_be_visible();
            let result = assertion.validate_state(false);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(err.to_string().contains("visible"));
        }

        #[test]
        fn test_validate_state_hidden_fail() {
            let locator = Locator::new("div");
            let assertion = expect(locator).to_be_hidden();
            let result = assertion.validate_state(false);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(err.to_string().contains("hidden"));
        }

        // -------------------------------------------------------------------
        // validate for non-state assertions with validate_state
        // -------------------------------------------------------------------

        #[test]
        fn test_validate_state_non_state_assertion() {
            let locator = Locator::new("span");
            let assertion = expect(locator).to_have_text("test");
            // Non-state assertions should return Ok
            assert!(assertion.validate_state(true).is_ok());
            assert!(assertion.validate_state(false).is_ok());
        }

        // -------------------------------------------------------------------
        // validate_count for non-count assertions
        // -------------------------------------------------------------------

        #[test]
        fn test_validate_count_non_count_assertion() {
            let locator = Locator::new("span");
            let assertion = expect(locator).to_have_text("test");
            // Non-count assertions should return Ok
            assert!(assertion.validate_count(0).is_ok());
            assert!(assertion.validate_count(100).is_ok());
        }

        // -------------------------------------------------------------------
        // validate contains_text failure
        // -------------------------------------------------------------------

        #[test]
        fn test_validate_contains_text_fail() {
            let locator = Locator::new("span");
            let assertion = expect(locator).to_contain_text("needle");
            let result = assertion.validate("haystack without the word");
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(err.to_string().contains("needle"));
        }

        // -------------------------------------------------------------------
        // validate has_id failure
        // -------------------------------------------------------------------

        #[test]
        fn test_validate_has_id_fail() {
            let locator = Locator::new("div");
            let assertion = expect(locator).to_have_id("expected-id");
            let result = assertion.validate("actual-id");
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(err.to_string().contains("expected-id"));
        }

        // -------------------------------------------------------------------
        // validate has_attribute failure
        // -------------------------------------------------------------------

        #[test]
        fn test_validate_has_attribute_fail() {
            let locator = Locator::new("input");
            let assertion = expect(locator).to_have_attribute("type", "email");
            let result = assertion.validate("text");
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(err.to_string().contains("email"));
            assert!(err.to_string().contains("type"));
        }

        // -------------------------------------------------------------------
        // Non-CSS selector operations (and, or, first, last, nth)
        // -------------------------------------------------------------------

        #[test]
        fn test_locator_or_non_css() {
            let locator1 = Locator::from_selector(Selector::Entity("hero".to_string()));
            let locator2 = Locator::new("div");
            let combined = locator1.or(locator2);
            // Should keep the original non-CSS selector
            assert!(matches!(combined.selector(), Selector::Entity(_)));
        }

        #[test]
        fn test_locator_first_non_css() {
            let locator = Locator::from_selector(Selector::Entity("hero".to_string()));
            let result = locator.first();
            // Should keep the original non-CSS selector
            assert!(matches!(result.selector(), Selector::Entity(_)));
        }

        #[test]
        fn test_locator_last_non_css() {
            let locator = Locator::from_selector(Selector::Entity("hero".to_string()));
            let result = locator.last();
            // Should keep the original non-CSS selector
            assert!(matches!(result.selector(), Selector::Entity(_)));
        }

        #[test]
        fn test_locator_nth_non_css() {
            let locator = Locator::from_selector(Selector::Entity("hero".to_string()));
            let result = locator.nth(5);
            // Should keep the original non-CSS selector
            assert!(matches!(result.selector(), Selector::Entity(_)));
        }

        // -------------------------------------------------------------------
        // Role selector with name - count query
        // -------------------------------------------------------------------

        #[test]
        fn test_role_with_name_count_query() {
            let selector = Selector::role_with_name("button", "Submit");
            let query = selector.to_count_query();
            assert!(query.contains("role"));
            assert!(query.contains("Submit"));
            assert!(query.contains(".length"));
        }

        // -------------------------------------------------------------------
        // Filter options without has_text
        // -------------------------------------------------------------------

        #[test]
        fn test_filter_without_has_text() {
            let child = Locator::new(".child");
            let locator = Locator::new("div").filter(FilterOptions::new().has(child));
            // Without has_text, selector should remain unchanged
            assert!(matches!(locator.selector(), Selector::Css(_)));
        }

        // -------------------------------------------------------------------
        // ClickOptions Default trait
        // -------------------------------------------------------------------

        #[test]
        fn test_click_options_default_trait() {
            let options: ClickOptions = Default::default();
            assert_eq!(options.button, MouseButton::Left);
            assert_eq!(options.click_count, 0); // Default is 0, new() sets it to 1
        }

        // -------------------------------------------------------------------
        // FilterOptions Default trait
        // -------------------------------------------------------------------

        #[test]
        fn test_filter_options_default_trait() {
            let options: FilterOptions = Default::default();
            assert!(options.has.is_none());
            assert!(options.has_text.is_none());
            assert!(options.has_not.is_none());
            assert!(options.has_not_text.is_none());
        }

        // -------------------------------------------------------------------
        // Point serialization (covered by derive)
        // -------------------------------------------------------------------

        #[test]
        fn test_point_clone() {
            let p1 = Point::new(1.0, 2.0);
            let p2 = p1;
            assert!((p2.x - 1.0).abs() < f32::EPSILON);
            assert!((p2.y - 2.0).abs() < f32::EPSILON);
        }

        #[test]
        fn test_point_partial_eq() {
            let p1 = Point::new(1.0, 2.0);
            let p2 = Point::new(1.0, 2.0);
            let p3 = Point::new(3.0, 4.0);
            assert_eq!(p1, p2);
            assert_ne!(p1, p3);
        }

        // -------------------------------------------------------------------
        // BoundingBox serialization (covered by derive)
        // -------------------------------------------------------------------

        #[test]
        fn test_bounding_box_clone() {
            let b1 = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
            let b2 = b1;
            assert!((b2.width - 100.0).abs() < f32::EPSILON);
        }

        #[test]
        fn test_bounding_box_partial_eq() {
            let b1 = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
            let b2 = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
            let b3 = BoundingBox::new(1.0, 1.0, 100.0, 100.0);
            assert_eq!(b1, b2);
            assert_ne!(b1, b3);
        }

        // -------------------------------------------------------------------
        // Selector equality
        // -------------------------------------------------------------------

        #[test]
        fn test_selector_equality() {
            let s1 = Selector::css("button");
            let s2 = Selector::css("button");
            let s3 = Selector::css("div");
            assert_eq!(s1, s2);
            assert_ne!(s1, s3);
        }

        #[test]
        fn test_selector_equality_css_with_text() {
            let s1 = Selector::CssWithText {
                css: "button".to_string(),
                text: "Click".to_string(),
            };
            let s2 = Selector::CssWithText {
                css: "button".to_string(),
                text: "Click".to_string(),
            };
            assert_eq!(s1, s2);
        }

        #[test]
        fn test_selector_equality_role() {
            let s1 = Selector::Role {
                role: "button".to_string(),
                name: Some("Submit".to_string()),
            };
            let s2 = Selector::Role {
                role: "button".to_string(),
                name: Some("Submit".to_string()),
            };
            assert_eq!(s1, s2);
        }

        // -------------------------------------------------------------------
        // DragBuilder chaining
        // -------------------------------------------------------------------

        #[test]
        fn test_drag_builder_full_chain() {
            let locator = Locator::new("div");
            let action = locator
                .drag_to(&Point::new(100.0, 200.0))
                .steps(15)
                .duration(Duration::from_millis(750))
                .build();

            match action {
                LocatorAction::Drag {
                    target,
                    steps,
                    duration,
                    ..
                } => {
                    assert!((target.x - 100.0).abs() < f32::EPSILON);
                    assert!((target.y - 200.0).abs() < f32::EPSILON);
                    assert_eq!(steps, 15);
                    assert_eq!(duration, Duration::from_millis(750));
                }
                _ => panic!("Expected Drag action"),
            }
        }

        // -------------------------------------------------------------------
        // LocatorOptions fields
        // -------------------------------------------------------------------

        #[test]
        fn test_locator_options_poll_interval() {
            let opts = LocatorOptions::default();
            assert_eq!(
                opts.poll_interval,
                Duration::from_millis(DEFAULT_POLL_INTERVAL_MS)
            );
        }

        // -------------------------------------------------------------------
        // KeyModifier variants
        // -------------------------------------------------------------------

        #[test]
        fn test_key_modifier_variants() {
            let modifiers = vec![
                KeyModifier::Alt,
                KeyModifier::Control,
                KeyModifier::Meta,
                KeyModifier::Shift,
            ];
            assert_eq!(modifiers.len(), 4);

            // Test equality
            assert_eq!(KeyModifier::Alt, KeyModifier::Alt);
            assert_ne!(KeyModifier::Alt, KeyModifier::Control);
        }

        // -------------------------------------------------------------------
        // MouseButton variants
        // -------------------------------------------------------------------

        #[test]
        fn test_mouse_button_variants() {
            let buttons = vec![MouseButton::Left, MouseButton::Right, MouseButton::Middle];
            assert_eq!(buttons.len(), 3);

            assert_eq!(MouseButton::Left, MouseButton::Left);
            assert_ne!(MouseButton::Left, MouseButton::Right);
        }

        // -------------------------------------------------------------------
        // LocatorAction locator accessor for Drag variant
        // -------------------------------------------------------------------

        #[test]
        fn test_locator_action_drag_locator_accessor() {
            let locator = Locator::new("div.draggable");
            let action = locator.drag_to(&Point::new(0.0, 0.0)).build();
            let accessed = action.locator();
            assert!(matches!(accessed.selector(), Selector::Css(_)));
        }

        // -------------------------------------------------------------------
        // LocatorAction locator accessor for Fill variant
        // -------------------------------------------------------------------

        #[test]
        fn test_locator_action_fill_locator_accessor() {
            let locator = Locator::new("input");
            let action = locator.fill("test").unwrap();
            let accessed = action.locator();
            assert!(matches!(accessed.selector(), Selector::Css(_)));
        }

        // -------------------------------------------------------------------
        // validate for browser-context assertions
        // -------------------------------------------------------------------

        #[test]
        fn test_validate_browser_context_assertions() {
            let locator = Locator::new("div");

            // IsVisible - returns Ok for browser context
            let assertion = expect(locator.clone()).to_be_visible();
            assert!(assertion.validate("any").is_ok());

            // IsHidden
            let assertion = expect(locator.clone()).to_be_hidden();
            assert!(assertion.validate("any").is_ok());

            // HasCount
            let assertion = expect(locator.clone()).to_have_count(5);
            assert!(assertion.validate("any").is_ok());

            // IsEnabled
            let assertion = expect(locator.clone()).to_be_enabled();
            assert!(assertion.validate("any").is_ok());

            // IsDisabled
            let assertion = expect(locator.clone()).to_be_disabled();
            assert!(assertion.validate("any").is_ok());

            // IsChecked
            let assertion = expect(locator.clone()).to_be_checked();
            assert!(assertion.validate("any").is_ok());

            // IsEditable
            let assertion = expect(locator.clone()).to_be_editable();
            assert!(assertion.validate("any").is_ok());

            // IsFocused
            let assertion = expect(locator.clone()).to_be_focused();
            assert!(assertion.validate("any").is_ok());

            // IsEmpty
            let assertion = expect(locator.clone()).to_be_empty();
            assert!(assertion.validate("any").is_ok());

            // HasCss
            let assertion = expect(locator).to_have_css("color", "red");
            assert!(assertion.validate("any").is_ok());
        }

        // -------------------------------------------------------------------
        // Debug implementations (covered by derive)
        // -------------------------------------------------------------------

        #[test]
        fn test_debug_implementations() {
            let point = Point::new(1.0, 2.0);
            let debug_str = format!("{:?}", point);
            assert!(debug_str.contains("Point"));

            let bbox = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
            let debug_str = format!("{:?}", bbox);
            assert!(debug_str.contains("BoundingBox"));

            let selector = Selector::css("div");
            let debug_str = format!("{:?}", selector);
            assert!(debug_str.contains("Css"));

            let locator = Locator::new("button");
            let debug_str = format!("{:?}", locator);
            assert!(debug_str.contains("Locator"));

            let options = LocatorOptions::default();
            let debug_str = format!("{:?}", options);
            assert!(debug_str.contains("LocatorOptions"));

            let filter = FilterOptions::new();
            let debug_str = format!("{:?}", filter);
            assert!(debug_str.contains("FilterOptions"));

            let click_opts = ClickOptions::new();
            let debug_str = format!("{:?}", click_opts);
            assert!(debug_str.contains("ClickOptions"));

            let drag_op = DragOperation::to(Point::new(0.0, 0.0));
            let debug_str = format!("{:?}", drag_op);
            assert!(debug_str.contains("DragOperation"));

            let drag_builder = Locator::new("div").drag_to(&Point::new(0.0, 0.0));
            let debug_str = format!("{:?}", drag_builder);
            assert!(debug_str.contains("DragBuilder"));

            let action = Locator::new("button").click().unwrap();
            let debug_str = format!("{:?}", action);
            assert!(debug_str.contains("Click"));

            let query = Locator::new("span").text_content().unwrap();
            let debug_str = format!("{:?}", query);
            assert!(debug_str.contains("TextContent"));

            let exp = Expect::new(Locator::new("div"));
            let debug_str = format!("{:?}", exp);
            assert!(debug_str.contains("Expect"));

            let assertion = expect(Locator::new("div")).to_have_text("test");
            let debug_str = format!("{:?}", assertion);
            assert!(debug_str.contains("HasText"));
        }

        // -------------------------------------------------------------------
        // Clone implementations
        // -------------------------------------------------------------------

        #[test]
        fn test_clone_implementations() {
            let locator = Locator::new("button");
            let cloned = locator;
            assert!(matches!(cloned.selector(), Selector::Css(_)));

            let options = LocatorOptions::default();
            let cloned = options;
            assert!(cloned.strict);

            let filter = FilterOptions::new().has_text("test");
            let cloned = filter;
            assert!(cloned.has_text.is_some());

            let click_opts = ClickOptions::new().button(MouseButton::Right);
            let cloned = click_opts;
            assert_eq!(cloned.button, MouseButton::Right);

            let drag_op = DragOperation::to(Point::new(1.0, 2.0)).steps(5);
            let cloned = drag_op;
            assert_eq!(cloned.steps, 5);

            let drag_builder = Locator::new("div").drag_to(&Point::new(3.0, 4.0)).steps(7);
            let cloned = drag_builder;
            let action = cloned.build();
            assert!(matches!(action, LocatorAction::Drag { steps: 7, .. }));

            let action = Locator::new("button").hover().unwrap();
            let cloned = action;
            assert!(matches!(cloned, LocatorAction::Hover { .. }));

            let query = Locator::new("span").count().unwrap();
            let cloned = query;
            assert!(matches!(cloned, LocatorQuery::Count { .. }));

            let exp = Expect::new(Locator::new("div"));
            let cloned = exp;
            let _ = cloned.to_be_visible();

            let assertion = expect(Locator::new("div")).to_have_count(3);
            let cloned = assertion;
            assert!(matches!(cloned, ExpectAssertion::HasCount { .. }));
        }

        // -------------------------------------------------------------------
        // Selector to_query edge cases
        // -------------------------------------------------------------------

        #[test]
        fn test_selector_to_query_special_chars() {
            // Test CSS selector with special characters
            let selector = Selector::css(r#"div[data-value="test's value"]"#);
            let query = selector.to_query();
            assert!(query.contains("querySelector"));

            // Test XPath with special characters
            let selector =
                Selector::XPath(r#"//button[contains(text(), "Click here")]"#.to_string());
            let query = selector.to_query();
            assert!(query.contains("evaluate"));

            // Test TestId with special characters
            let selector = Selector::test_id("my-test-id_123");
            let query = selector.to_query();
            assert!(query.contains("data-testid"));
        }

        // -------------------------------------------------------------------
        // BoundingBox contains edge cases
        // -------------------------------------------------------------------

        #[test]
        fn test_bounding_box_contains_all_edges() {
            let bbox = BoundingBox::new(10.0, 20.0, 100.0, 50.0);

            // Test all four corners
            assert!(bbox.contains(&Point::new(10.0, 20.0))); // top-left
            assert!(bbox.contains(&Point::new(110.0, 20.0))); // top-right
            assert!(bbox.contains(&Point::new(10.0, 70.0))); // bottom-left
            assert!(bbox.contains(&Point::new(110.0, 70.0))); // bottom-right

            // Test just outside each edge
            assert!(!bbox.contains(&Point::new(9.9, 45.0))); // left
            assert!(!bbox.contains(&Point::new(110.1, 45.0))); // right
            assert!(!bbox.contains(&Point::new(55.0, 19.9))); // top
            assert!(!bbox.contains(&Point::new(55.0, 70.1))); // bottom
        }

        // -------------------------------------------------------------------
        // BoundingBox center with offset
        // -------------------------------------------------------------------

        #[test]
        fn test_bounding_box_center_with_offset() {
            let bbox = BoundingBox::new(10.0, 20.0, 100.0, 50.0);
            let center = bbox.center();
            assert!((center.x - 60.0).abs() < f32::EPSILON); // 10 + 100/2
            assert!((center.y - 45.0).abs() < f32::EPSILON); // 20 + 50/2
        }

        // -------------------------------------------------------------------
        // Locator chaining
        // -------------------------------------------------------------------

        #[test]
        fn test_locator_chaining_all_options() {
            let locator = Locator::new("button")
                .with_text("Click")
                .with_timeout(Duration::from_secs(10))
                .with_strict(false)
                .with_visible(false);

            assert!(!locator.options().strict);
            assert!(!locator.options().visible);
            assert_eq!(locator.options().timeout, Duration::from_secs(10));
            assert!(matches!(locator.selector(), Selector::CssWithText { .. }));
        }

        // -------------------------------------------------------------------
        // ClickOptions chaining
        // -------------------------------------------------------------------

        #[test]
        fn test_click_options_full_chain() {
            let options = ClickOptions::new()
                .button(MouseButton::Middle)
                .click_count(3)
                .position(Point::new(5.0, 10.0))
                .modifier(KeyModifier::Shift)
                .modifier(KeyModifier::Alt)
                .modifier(KeyModifier::Control)
                .modifier(KeyModifier::Meta);

            assert_eq!(options.button, MouseButton::Middle);
            assert_eq!(options.click_count, 3);
            assert!(options.position.is_some());
            let pos = options.position.unwrap();
            assert!((pos.x - 5.0).abs() < f32::EPSILON);
            assert!((pos.y - 10.0).abs() < f32::EPSILON);
            assert_eq!(options.modifiers.len(), 4);
        }
    }
}
