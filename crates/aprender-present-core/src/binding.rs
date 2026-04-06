//! Interactive state binding for reactive UI.
//!
//! This module provides mechanisms to bind UI widget properties to application
//! state, enabling declarative data flow and automatic UI updates.
//!
//! # Binding Types
//!
//! - `Binding<T>` - Two-way binding for read/write access
//! - `Derived<T>` - Read-only computed value from state
//! - `PropertyPath` - Path to a property in state (e.g., "user.name")
//!
//! # Example
//!
//! ```ignore
//! use presentar_core::binding::{Binding, Derived, PropertyPath};
//!
//! // Create a binding to state.count
//! let count_binding = Binding::new(|| state.count, |v| state.count = v);
//!
//! // Create a derived value
//! let doubled = Derived::new(|| state.count * 2);
//! ```

use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fmt;
use std::sync::{Arc, RwLock};

/// Type alias for subscriber callbacks.
type SubscriberFn<T> = Box<dyn Fn(&T) + Send + Sync>;

/// Type alias for subscribers list.
type Subscribers<T> = Arc<RwLock<Vec<SubscriberFn<T>>>>;

/// A property path for accessing nested state.
///
/// Property paths use dot notation: "user.profile.name"
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PropertyPath {
    segments: Vec<String>,
}

impl PropertyPath {
    /// Create a new property path from a string.
    #[must_use]
    pub fn new(path: &str) -> Self {
        let segments = path
            .split('.')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        Self { segments }
    }

    /// Create an empty root path.
    #[must_use]
    pub const fn root() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    /// Get path segments.
    #[must_use]
    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    /// Check if path is empty (root).
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.segments.is_empty()
    }

    /// Get the number of segments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Check if path is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Append a segment to the path.
    #[must_use]
    pub fn join(&self, segment: &str) -> Self {
        let mut segments = self.segments.clone();
        segments.push(segment.to_string());
        Self { segments }
    }

    /// Get the parent path.
    #[must_use]
    pub fn parent(&self) -> Option<Self> {
        if self.segments.is_empty() {
            None
        } else {
            let mut segments = self.segments.clone();
            segments.pop();
            Some(Self { segments })
        }
    }

    /// Get the last segment (leaf name).
    #[must_use]
    pub fn leaf(&self) -> Option<&str> {
        self.segments.last().map(String::as_str)
    }

    /// Convert to string representation.
    #[must_use]
    pub fn to_string_path(&self) -> String {
        self.segments.join(".")
    }
}

impl fmt::Display for PropertyPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_path())
    }
}

impl From<&str> for PropertyPath {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

/// Binding direction for property connections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BindingDirection {
    /// One-way binding: state → widget
    #[default]
    OneWay,
    /// Two-way binding: state ↔ widget
    TwoWay,
    /// One-time binding: state → widget (initial only)
    OneTime,
}

/// A binding configuration for connecting state to widget properties.
#[derive(Debug, Clone)]
pub struct BindingConfig {
    /// Source path in state
    pub source: PropertyPath,
    /// Target property on widget
    pub target: String,
    /// Binding direction
    pub direction: BindingDirection,
    /// Optional transform function name
    pub transform: Option<String>,
    /// Optional fallback value (as string)
    pub fallback: Option<String>,
}

impl BindingConfig {
    /// Create a new one-way binding.
    #[must_use]
    pub fn one_way(source: impl Into<PropertyPath>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            direction: BindingDirection::OneWay,
            transform: None,
            fallback: None,
        }
    }

    /// Create a new two-way binding.
    #[must_use]
    pub fn two_way(source: impl Into<PropertyPath>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            direction: BindingDirection::TwoWay,
            transform: None,
            fallback: None,
        }
    }

    /// Set a transform function.
    #[must_use]
    pub fn transform(mut self, name: impl Into<String>) -> Self {
        self.transform = Some(name.into());
        self
    }

    /// Set a fallback value.
    #[must_use]
    pub fn fallback(mut self, value: impl Into<String>) -> Self {
        self.fallback = Some(value.into());
        self
    }
}

/// Trait for types that can be bound to state.
pub trait Bindable: Any + Send + Sync {
    /// Get bindings for this widget.
    fn bindings(&self) -> Vec<BindingConfig>;

    /// Set bindings for this widget.
    fn set_bindings(&mut self, bindings: Vec<BindingConfig>);

    /// Apply a binding value update.
    fn apply_binding(&mut self, target: &str, value: &dyn Any) -> bool;

    /// Get the current value for a binding target.
    fn get_binding_value(&self, target: &str) -> Option<Box<dyn Any + Send>>;
}

/// A reactive cell that holds a value and notifies on changes.
pub struct ReactiveCell<T> {
    value: Arc<RwLock<T>>,
    subscribers: Subscribers<T>,
}

impl<T: Clone + Send + Sync + 'static> ReactiveCell<T> {
    /// Create a new reactive cell with an initial value.
    pub fn new(value: T) -> Self {
        Self {
            value: Arc::new(RwLock::new(value)),
            subscribers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Get the current value.
    pub fn get(&self) -> T {
        self.value
            .read()
            .expect("ReactiveCell lock poisoned")
            .clone()
    }

    /// Set a new value, notifying subscribers.
    pub fn set(&self, value: T) {
        {
            let mut guard = self.value.write().expect("ReactiveCell lock poisoned");
            *guard = value;
        }
        self.notify();
    }

    /// Update the value using a function.
    pub fn update<F>(&self, f: F)
    where
        F: FnOnce(&mut T),
    {
        {
            let mut guard = self.value.write().expect("ReactiveCell lock poisoned");
            f(&mut guard);
        }
        self.notify();
    }

    /// Subscribe to value changes.
    pub fn subscribe<F>(&self, callback: F)
    where
        F: Fn(&T) + Send + Sync + 'static,
    {
        self.subscribers
            .write()
            .expect("ReactiveCell lock poisoned")
            .push(Box::new(callback));
    }

    fn notify(&self) {
        let value = self.value.read().expect("ReactiveCell lock poisoned");
        let subscribers = self.subscribers.read().expect("ReactiveCell lock poisoned");
        for sub in subscribers.iter() {
            sub(&value);
        }
    }
}

impl<T: Clone + Send + Sync> Clone for ReactiveCell<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            subscribers: Arc::new(RwLock::new(Vec::new())), // Don't clone subscribers
        }
    }
}

impl<T: Clone + Send + Sync + Default + 'static> Default for ReactiveCell<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: Clone + Send + Sync + fmt::Debug + 'static> fmt::Debug for ReactiveCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReactiveCell")
            .field(
                "value",
                &*self.value.read().expect("ReactiveCell lock poisoned"),
            )
            .finish_non_exhaustive()
    }
}

/// A computed value derived from other reactive sources.
pub struct Computed<T> {
    #[allow(dead_code)]
    compute: Box<dyn Fn() -> T + Send + Sync>,
    cached: Arc<RwLock<Option<T>>>,
    dirty: Arc<RwLock<bool>>,
}

impl<T: Clone + Send + Sync + 'static> Computed<T> {
    /// Create a new computed value.
    pub fn new<F>(compute: F) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        Self {
            compute: Box::new(compute),
            cached: Arc::new(RwLock::new(None)),
            dirty: Arc::new(RwLock::new(true)),
        }
    }

    /// Get the computed value (caches result).
    pub fn get(&self) -> T {
        let dirty = *self.dirty.read().expect("Computed lock poisoned");
        if dirty {
            let value = (self.compute)();
            *self.cached.write().expect("Computed lock poisoned") = Some(value.clone());
            *self.dirty.write().expect("Computed lock poisoned") = false;
            value
        } else {
            self.cached
                .read()
                .expect("Computed lock poisoned")
                .clone()
                .expect("Computed cache should contain value when not dirty")
        }
    }

    /// Mark the computed value as dirty (needs recomputation).
    pub fn invalidate(&self) {
        *self.dirty.write().expect("Computed lock poisoned") = true;
    }
}

impl<T: Clone + Send + Sync + fmt::Debug + 'static> fmt::Debug for Computed<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Computed")
            .field(
                "cached",
                &*self.cached.read().expect("Computed lock poisoned"),
            )
            .field(
                "dirty",
                &*self.dirty.read().expect("Computed lock poisoned"),
            )
            .finish_non_exhaustive()
    }
}

/// A binding expression that can be evaluated against state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingExpression {
    /// Expression string (e.g., "{{ user.name }}" or "{{ count * 2 }}")
    pub expression: String,
    /// Parsed dependencies (property paths used)
    pub dependencies: Vec<PropertyPath>,
}

impl BindingExpression {
    /// Create a new binding expression.
    #[must_use]
    pub fn new(expression: impl Into<String>) -> Self {
        let expression = expression.into();
        let dependencies = Self::parse_dependencies(&expression);
        Self {
            expression,
            dependencies,
        }
    }

    /// Create a simple property binding.
    #[must_use]
    pub fn property(path: impl Into<PropertyPath>) -> Self {
        let path: PropertyPath = path.into();
        let expression = format!("{{{{ {} }}}}", path.to_string_path());
        Self {
            expression,
            dependencies: vec![path],
        }
    }

    /// Check if this is a simple property binding (no transforms).
    #[must_use]
    pub fn is_simple_property(&self) -> bool {
        self.dependencies.len() == 1
            && self.expression.trim().starts_with("{{")
            && self.expression.trim().ends_with("}}")
    }

    /// Get the property path if this is a simple binding.
    #[must_use]
    pub fn as_property(&self) -> Option<&PropertyPath> {
        if self.is_simple_property() {
            self.dependencies.first()
        } else {
            None
        }
    }

    fn parse_dependencies(expression: &str) -> Vec<PropertyPath> {
        let mut deps = Vec::new();
        let mut in_binding = false;
        let mut current = String::new();

        let chars: Vec<char> = expression.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if i + 1 < chars.len() && chars[i] == '{' && chars[i + 1] == '{' {
                in_binding = true;
                i += 2;
                continue;
            }

            if i + 1 < chars.len() && chars[i] == '}' && chars[i + 1] == '}' {
                if !current.is_empty() {
                    // Extract property path from current (may have transforms)
                    let path_str = current.split('|').next().unwrap_or("").trim();
                    if !path_str.is_empty() && !path_str.contains(|c: char| c.is_whitespace()) {
                        deps.push(PropertyPath::new(path_str));
                    }
                    current.clear();
                }
                in_binding = false;
                i += 2;
                continue;
            }

            if in_binding {
                current.push(chars[i]);
            }

            i += 1;
        }

        deps
    }
}

/// Event binding that maps widget events to state messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventBinding {
    /// Widget event name (e.g., "click", "change", "submit")
    pub event: String,
    /// Action to dispatch
    pub action: ActionBinding,
}

/// Action binding for state updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionBinding {
    /// Set a property to a value
    SetProperty {
        /// Property path
        path: PropertyPath,
        /// Value expression
        value: String,
    },
    /// Toggle a boolean property
    ToggleProperty {
        /// Property path
        path: PropertyPath,
    },
    /// Increment a numeric property
    IncrementProperty {
        /// Property path
        path: PropertyPath,
        /// Amount to increment (default 1)
        amount: Option<f64>,
    },
    /// Navigate to a route
    Navigate {
        /// Route path
        route: String,
    },
    /// Dispatch a custom message
    Dispatch {
        /// Message type/name
        message: String,
        /// Optional payload
        payload: Option<String>,
    },
    /// Execute multiple actions
    Batch {
        /// Actions to execute
        actions: Vec<Self>,
    },
}

impl EventBinding {
    /// Create a new event binding.
    #[must_use]
    pub fn new(event: impl Into<String>, action: ActionBinding) -> Self {
        Self {
            event: event.into(),
            action,
        }
    }

    /// Create a click event binding.
    #[must_use]
    pub fn on_click(action: ActionBinding) -> Self {
        Self::new("click", action)
    }

    /// Create a change event binding.
    #[must_use]
    pub fn on_change(action: ActionBinding) -> Self {
        Self::new("change", action)
    }
}

impl ActionBinding {
    /// Create a set property action.
    #[must_use]
    pub fn set(path: impl Into<PropertyPath>, value: impl Into<String>) -> Self {
        Self::SetProperty {
            path: path.into(),
            value: value.into(),
        }
    }

    /// Create a toggle action.
    #[must_use]
    pub fn toggle(path: impl Into<PropertyPath>) -> Self {
        Self::ToggleProperty { path: path.into() }
    }

    /// Create an increment action.
    #[must_use]
    pub fn increment(path: impl Into<PropertyPath>) -> Self {
        Self::IncrementProperty {
            path: path.into(),
            amount: None,
        }
    }

    /// Create an increment by amount action.
    #[must_use]
    pub fn increment_by(path: impl Into<PropertyPath>, amount: f64) -> Self {
        Self::IncrementProperty {
            path: path.into(),
            amount: Some(amount),
        }
    }

    /// Create a navigate action.
    #[must_use]
    pub fn navigate(route: impl Into<String>) -> Self {
        Self::Navigate {
            route: route.into(),
        }
    }

    /// Create a dispatch action.
    #[must_use]
    pub fn dispatch(message: impl Into<String>) -> Self {
        Self::Dispatch {
            message: message.into(),
            payload: None,
        }
    }

    /// Create a dispatch with payload action.
    #[must_use]
    pub fn dispatch_with(message: impl Into<String>, payload: impl Into<String>) -> Self {
        Self::Dispatch {
            message: message.into(),
            payload: Some(payload.into()),
        }
    }

    /// Create a batch of actions.
    #[must_use]
    pub fn batch(actions: impl IntoIterator<Item = Self>) -> Self {
        Self::Batch {
            actions: actions.into_iter().collect(),
        }
    }
}

// =============================================================================
// BindingManager - Orchestrates State-Widget Bindings
// =============================================================================

/// Manages bindings between application state and widget properties.
///
/// The `BindingManager` provides:
/// - Registration of two-way bindings
/// - Automatic propagation of state changes to widgets
/// - Handling of widget changes back to state
/// - Debouncing support for frequent updates
#[derive(Debug, Default)]
pub struct BindingManager {
    /// Active bindings
    bindings: Vec<ActiveBinding>,
    /// Whether to debounce updates
    debounce_ms: Option<u32>,
    /// Pending updates queue
    pending_updates: Vec<PendingUpdate>,
}

/// An active binding between state and widget.
#[derive(Debug, Clone)]
pub struct ActiveBinding {
    /// Unique binding ID
    pub id: BindingId,
    /// Widget ID
    pub widget_id: String,
    /// Binding configuration
    pub config: BindingConfig,
    /// Current state value (as string for simplicity)
    pub current_value: Option<String>,
    /// Whether binding is active
    pub active: bool,
}

/// Unique binding identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BindingId(pub u64);

/// A pending update to be applied.
#[derive(Debug, Clone)]
pub struct PendingUpdate {
    /// Source (widget or state)
    pub source: UpdateSource,
    /// Property path
    pub path: PropertyPath,
    /// New value as string
    pub value: String,
    /// Timestamp
    pub timestamp: u64,
}

/// Source of a binding update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateSource {
    /// Update from state
    State,
    /// Update from widget
    Widget,
}

impl BindingManager {
    /// Create a new binding manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set debounce delay in milliseconds.
    #[must_use]
    pub fn with_debounce(mut self, ms: u32) -> Self {
        self.debounce_ms = Some(ms);
        self
    }

    /// Register a binding between state and widget.
    pub fn register(&mut self, widget_id: impl Into<String>, config: BindingConfig) -> BindingId {
        let id = BindingId(self.bindings.len() as u64);
        self.bindings.push(ActiveBinding {
            id,
            widget_id: widget_id.into(),
            config,
            current_value: None,
            active: true,
        });
        id
    }

    /// Unregister a binding.
    pub fn unregister(&mut self, id: BindingId) {
        if let Some(binding) = self.bindings.iter_mut().find(|b| b.id == id) {
            binding.active = false;
        }
    }

    /// Get bindings for a widget.
    #[must_use]
    pub fn bindings_for_widget(&self, widget_id: &str) -> Vec<&ActiveBinding> {
        self.bindings
            .iter()
            .filter(|b| b.active && b.widget_id == widget_id)
            .collect()
    }

    /// Get bindings for a state path.
    #[must_use]
    pub fn bindings_for_path(&self, path: &PropertyPath) -> Vec<&ActiveBinding> {
        self.bindings
            .iter()
            .filter(|b| b.active && &b.config.source == path)
            .collect()
    }

    /// Handle state change, propagate to widgets.
    pub fn on_state_change(&mut self, path: &PropertyPath, value: &str) -> Vec<WidgetUpdate> {
        let mut updates = Vec::new();

        for binding in &mut self.bindings {
            if !binding.active {
                continue;
            }

            // Check if this path affects this binding
            if &binding.config.source == path
                || path
                    .to_string_path()
                    .starts_with(&binding.config.source.to_string_path())
            {
                binding.current_value = Some(value.to_string());

                updates.push(WidgetUpdate {
                    widget_id: binding.widget_id.clone(),
                    property: binding.config.target.clone(),
                    value: value.to_string(),
                });
            }
        }

        updates
    }

    /// Handle widget change, propagate to state.
    pub fn on_widget_change(
        &mut self,
        widget_id: &str,
        property: &str,
        value: &str,
    ) -> Vec<StateUpdate> {
        let mut updates = Vec::new();

        for binding in &self.bindings {
            if !binding.active {
                continue;
            }

            // Only two-way bindings propagate back to state
            if binding.config.direction != BindingDirection::TwoWay {
                continue;
            }

            if binding.widget_id == widget_id && binding.config.target == property {
                updates.push(StateUpdate {
                    path: binding.config.source.clone(),
                    value: value.to_string(),
                });
            }
        }

        updates
    }

    /// Queue an update (for debouncing).
    pub fn queue_update(&mut self, source: UpdateSource, path: PropertyPath, value: String) {
        self.pending_updates.push(PendingUpdate {
            source,
            path,
            value,
            timestamp: 0, // Would be set to actual timestamp
        });
    }

    /// Flush pending updates.
    pub fn flush(&mut self) -> (Vec<WidgetUpdate>, Vec<StateUpdate>) {
        let mut widget_updates = Vec::new();
        let mut state_updates = Vec::new();

        // Drain into separate Vec to avoid borrow issues
        let updates: Vec<PendingUpdate> = self.pending_updates.drain(..).collect();

        for update in updates {
            match update.source {
                UpdateSource::State => {
                    widget_updates.extend(self.on_state_change(&update.path, &update.value));
                }
                UpdateSource::Widget => {
                    // For widget updates, we'd need widget_id context
                    state_updates.push(StateUpdate {
                        path: update.path,
                        value: update.value,
                    });
                }
            }
        }

        (widget_updates, state_updates)
    }

    /// Get number of active bindings.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.bindings.iter().filter(|b| b.active).count()
    }

    /// Clear all bindings.
    pub fn clear(&mut self) {
        self.bindings.clear();
        self.pending_updates.clear();
    }
}

/// Update to apply to a widget.
#[derive(Debug, Clone)]
pub struct WidgetUpdate {
    /// Target widget ID
    pub widget_id: String,
    /// Property to update
    pub property: String,
    /// New value
    pub value: String,
}

/// Update to apply to state.
#[derive(Debug, Clone)]
pub struct StateUpdate {
    /// Property path
    pub path: PropertyPath,
    /// New value
    pub value: String,
}

// =============================================================================
// ValueConverter - Type Conversion for Bindings
// =============================================================================

/// Converts values between different types for binding.
pub trait ValueConverter: Send + Sync {
    /// Convert from source type to target type.
    fn convert(&self, value: &str) -> Result<String, ConversionError>;

    /// Convert back from target type to source type.
    fn convert_back(&self, value: &str) -> Result<String, ConversionError>;
}

/// Error during value conversion.
#[derive(Debug, Clone)]
pub struct ConversionError {
    /// Error message
    pub message: String,
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "conversion error: {}", self.message)
    }
}

impl std::error::Error for ConversionError {}

/// Identity converter (no conversion).
#[derive(Debug, Default)]
pub struct IdentityConverter;

impl ValueConverter for IdentityConverter {
    fn convert(&self, value: &str) -> Result<String, ConversionError> {
        Ok(value.to_string())
    }

    fn convert_back(&self, value: &str) -> Result<String, ConversionError> {
        Ok(value.to_string())
    }
}

/// Boolean to string converter.
#[derive(Debug, Default)]
pub struct BoolToStringConverter {
    /// String for true value
    pub true_string: String,
    /// String for false value
    pub false_string: String,
}

impl BoolToStringConverter {
    /// Create with "true"/"false" strings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            true_string: "true".to_string(),
            false_string: "false".to_string(),
        }
    }

    /// Create with custom strings.
    #[must_use]
    pub fn with_strings(true_str: impl Into<String>, false_str: impl Into<String>) -> Self {
        Self {
            true_string: true_str.into(),
            false_string: false_str.into(),
        }
    }
}

impl ValueConverter for BoolToStringConverter {
    fn convert(&self, value: &str) -> Result<String, ConversionError> {
        match value {
            "true" | "1" | "yes" => Ok(self.true_string.clone()),
            "false" | "0" | "no" => Ok(self.false_string.clone()),
            _ => Err(ConversionError {
                message: format!("cannot convert '{value}' to bool"),
            }),
        }
    }

    fn convert_back(&self, value: &str) -> Result<String, ConversionError> {
        if value == self.true_string {
            Ok("true".to_string())
        } else if value == self.false_string {
            Ok("false".to_string())
        } else {
            Err(ConversionError {
                message: format!("cannot convert '{value}' back to bool"),
            })
        }
    }
}

/// Number formatter converter.
#[derive(Debug, Default)]
pub struct NumberFormatConverter {
    /// Decimal places
    pub decimals: usize,
    /// Prefix (e.g., "$")
    pub prefix: String,
    /// Suffix (e.g., "%")
    pub suffix: String,
}

impl NumberFormatConverter {
    /// Create default formatter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set decimal places.
    #[must_use]
    pub fn decimals(mut self, places: usize) -> Self {
        self.decimals = places;
        self
    }

    /// Set prefix.
    #[must_use]
    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Set suffix.
    #[must_use]
    pub fn suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = suffix.into();
        self
    }
}

impl ValueConverter for NumberFormatConverter {
    fn convert(&self, value: &str) -> Result<String, ConversionError> {
        let num: f64 = value.parse().map_err(|_| ConversionError {
            message: format!("cannot parse '{value}' as number"),
        })?;

        let formatted = format!("{:.prec$}", num, prec = self.decimals);
        Ok(format!("{}{}{}", self.prefix, formatted, self.suffix))
    }

    fn convert_back(&self, value: &str) -> Result<String, ConversionError> {
        // Strip prefix and suffix
        let stripped = value
            .strip_prefix(&self.prefix)
            .unwrap_or(value)
            .strip_suffix(&self.suffix)
            .unwrap_or(value)
            .trim();

        // Validate it's a number
        let _: f64 = stripped.parse().map_err(|_| ConversionError {
            message: format!("cannot parse '{stripped}' as number"),
        })?;

        Ok(stripped.to_string())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
#[path = "binding_tests.rs"]
mod tests;
