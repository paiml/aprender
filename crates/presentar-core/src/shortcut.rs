#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! Keyboard shortcut management system.
//!
//! This module provides:
//! - Keyboard shortcut registration and handling
//! - Modifier key support (Ctrl, Alt, Shift, Meta)
//! - Context-aware shortcuts (global, focused widget, etc.)
//! - Shortcut conflict detection

use crate::event::Key;
use crate::widget::WidgetId;
use std::collections::HashMap;

/// Modifier keys for keyboard shortcuts.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub struct Modifiers {
    /// Control key (Cmd on Mac).
    pub ctrl: bool,
    /// Alt key (Option on Mac).
    pub alt: bool,
    /// Shift key.
    pub shift: bool,
    /// Meta key (Windows key, Cmd on Mac).
    pub meta: bool,
}

impl Modifiers {
    /// No modifiers.
    pub const NONE: Self = Self {
        ctrl: false,
        alt: false,
        shift: false,
        meta: false,
    };

    /// Ctrl only.
    pub const CTRL: Self = Self {
        ctrl: true,
        alt: false,
        shift: false,
        meta: false,
    };

    /// Alt only.
    pub const ALT: Self = Self {
        ctrl: false,
        alt: true,
        shift: false,
        meta: false,
    };

    /// Shift only.
    pub const SHIFT: Self = Self {
        ctrl: false,
        alt: false,
        shift: true,
        meta: false,
    };

    /// Meta only.
    pub const META: Self = Self {
        ctrl: false,
        alt: false,
        shift: false,
        meta: true,
    };

    /// Ctrl+Shift.
    pub const CTRL_SHIFT: Self = Self {
        ctrl: true,
        alt: false,
        shift: true,
        meta: false,
    };

    /// Ctrl+Alt.
    pub const CTRL_ALT: Self = Self {
        ctrl: true,
        alt: true,
        shift: false,
        meta: false,
    };

    /// Create custom modifiers.
    pub const fn new(ctrl: bool, alt: bool, shift: bool, meta: bool) -> Self {
        Self {
            ctrl,
            alt,
            shift,
            meta,
        }
    }

    /// Check if any modifier is pressed.
    pub const fn any(&self) -> bool {
        self.ctrl || self.alt || self.shift || self.meta
    }

    /// Check if no modifier is pressed.
    pub const fn none(&self) -> bool {
        !self.any()
    }

    /// Get a display string for the modifiers.
    pub fn display(&self) -> String {
        let mut parts = Vec::new();
        if self.ctrl {
            parts.push("Ctrl");
        }
        if self.alt {
            parts.push("Alt");
        }
        if self.shift {
            parts.push("Shift");
        }
        if self.meta {
            parts.push("Meta");
        }
        parts.join("+")
    }
}

/// A keyboard shortcut (key + modifiers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shortcut {
    /// The key.
    pub key: Key,
    /// Modifier keys.
    pub modifiers: Modifiers,
}

impl Shortcut {
    /// Create a new shortcut.
    pub const fn new(key: Key, modifiers: Modifiers) -> Self {
        Self { key, modifiers }
    }

    /// Create a shortcut with no modifiers.
    pub const fn key(key: Key) -> Self {
        Self::new(key, Modifiers::NONE)
    }

    /// Create a shortcut with Ctrl modifier.
    pub const fn ctrl(key: Key) -> Self {
        Self::new(key, Modifiers::CTRL)
    }

    /// Create a shortcut with Alt modifier.
    pub const fn alt(key: Key) -> Self {
        Self::new(key, Modifiers::ALT)
    }

    /// Create a shortcut with Shift modifier.
    pub const fn shift(key: Key) -> Self {
        Self::new(key, Modifiers::SHIFT)
    }

    /// Create a shortcut with Ctrl+Shift modifiers.
    pub const fn ctrl_shift(key: Key) -> Self {
        Self::new(key, Modifiers::CTRL_SHIFT)
    }

    /// Get a display string for the shortcut.
    pub fn display(&self) -> String {
        let key_name = format!("{:?}", self.key);
        if self.modifiers.none() {
            key_name
        } else {
            format!("{}+{}", self.modifiers.display(), key_name)
        }
    }

    /// Common shortcuts
    pub const COPY: Self = Self::ctrl(Key::C);
    pub const CUT: Self = Self::ctrl(Key::X);
    pub const PASTE: Self = Self::ctrl(Key::V);
    pub const UNDO: Self = Self::ctrl(Key::Z);
    pub const REDO: Self = Self::ctrl_shift(Key::Z);
    pub const SAVE: Self = Self::ctrl(Key::S);
    pub const SELECT_ALL: Self = Self::ctrl(Key::A);
    pub const FIND: Self = Self::ctrl(Key::F);
    pub const ESCAPE: Self = Self::key(Key::Escape);
    pub const ENTER: Self = Self::key(Key::Enter);
    pub const TAB: Self = Self::key(Key::Tab);
    pub const DELETE: Self = Self::key(Key::Delete);
    pub const BACKSPACE: Self = Self::key(Key::Backspace);
}

/// Unique ID for a shortcut binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShortcutId(pub u64);

impl ShortcutId {
    /// Create a new shortcut ID.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Context in which a shortcut is active.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum ShortcutContext {
    /// Global - active everywhere.
    #[default]
    Global,
    /// Only when a specific widget has focus.
    Widget(WidgetId),
    /// Only when a widget type has focus.
    WidgetType(String),
    /// Custom context identified by name.
    Custom(String),
}

/// Priority for shortcut resolution when multiple match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum ShortcutPriority {
    /// Low priority - checked last.
    Low = 0,
    /// Normal priority.
    #[default]
    Normal = 1,
    /// High priority - checked first.
    High = 2,
}

/// Callback for shortcut handling.
pub type ShortcutHandler = Box<dyn FnMut() -> bool + Send>;

/// Registration for a shortcut binding.
struct ShortcutBinding {
    #[allow(dead_code)]
    id: ShortcutId,
    shortcut: Shortcut,
    context: ShortcutContext,
    priority: ShortcutPriority,
    description: String,
    enabled: bool,
}

/// Manager for keyboard shortcuts.
pub struct ShortcutManager {
    /// Next binding ID.
    next_id: u64,
    /// Registered bindings.
    bindings: HashMap<ShortcutId, ShortcutBinding>,
    /// Handlers (separate for mutability).
    handlers: HashMap<ShortcutId, ShortcutHandler>,
    /// Index by shortcut for fast lookup.
    by_shortcut: HashMap<Shortcut, Vec<ShortcutId>>,
    /// Current active contexts.
    active_contexts: Vec<ShortcutContext>,
    /// Current modifier state.
    modifiers: Modifiers,
}

impl ShortcutManager {
    /// Create a new shortcut manager.
    pub fn new() -> Self {
        Self {
            next_id: 0,
            bindings: HashMap::new(),
            handlers: HashMap::new(),
            by_shortcut: HashMap::new(),
            active_contexts: vec![ShortcutContext::Global],
            modifiers: Modifiers::NONE,
        }
    }

    /// Register a shortcut.
    pub fn register(&mut self, shortcut: Shortcut, handler: ShortcutHandler) -> ShortcutId {
        self.register_with_options(
            shortcut,
            handler,
            ShortcutContext::Global,
            ShortcutPriority::Normal,
            "",
        )
    }

    /// Register a shortcut with full options.
    pub fn register_with_options(
        &mut self,
        shortcut: Shortcut,
        handler: ShortcutHandler,
        context: ShortcutContext,
        priority: ShortcutPriority,
        description: &str,
    ) -> ShortcutId {
        let id = ShortcutId::new(self.next_id);
        self.next_id += 1;

        let binding = ShortcutBinding {
            id,
            shortcut,
            context,
            priority,
            description: description.to_string(),
            enabled: true,
        };

        self.bindings.insert(id, binding);
        self.handlers.insert(id, handler);

        self.by_shortcut.entry(shortcut).or_default().push(id);

        id
    }

    /// Unregister a shortcut.
    pub fn unregister(&mut self, id: ShortcutId) -> bool {
        if let Some(binding) = self.bindings.remove(&id) {
            self.handlers.remove(&id);

            if let Some(ids) = self.by_shortcut.get_mut(&binding.shortcut) {
                ids.retain(|&i| i != id);
            }

            true
        } else {
            false
        }
    }

    /// Enable or disable a shortcut.
    pub fn set_enabled(&mut self, id: ShortcutId, enabled: bool) {
        if let Some(binding) = self.bindings.get_mut(&id) {
            binding.enabled = enabled;
        }
    }

    /// Check if a shortcut is enabled.
    pub fn is_enabled(&self, id: ShortcutId) -> bool {
        self.bindings.get(&id).is_some_and(|b| b.enabled)
    }

    /// Set the current modifier state.
    pub fn set_modifiers(&mut self, modifiers: Modifiers) {
        self.modifiers = modifiers;
    }

    /// Get the current modifier state.
    pub fn modifiers(&self) -> Modifiers {
        self.modifiers
    }

    /// Push an active context.
    pub fn push_context(&mut self, context: ShortcutContext) {
        self.active_contexts.push(context);
    }

    /// Pop the most recent context.
    pub fn pop_context(&mut self) -> Option<ShortcutContext> {
        if self.active_contexts.len() > 1 {
            self.active_contexts.pop()
        } else {
            None
        }
    }

    /// Set the focused widget context.
    pub fn set_focused_widget(&mut self, widget_id: Option<WidgetId>) {
        // Remove any existing widget context
        self.active_contexts
            .retain(|c| !matches!(c, ShortcutContext::Widget(_)));

        if let Some(id) = widget_id {
            self.active_contexts.push(ShortcutContext::Widget(id));
        }
    }

    /// Handle a key press and trigger matching shortcuts.
    /// Returns true if a shortcut was triggered.
    pub fn handle_key(&mut self, key: Key) -> bool {
        let shortcut = Shortcut::new(key, self.modifiers);
        self.trigger(shortcut)
    }

    /// Trigger a shortcut directly.
    pub fn trigger(&mut self, shortcut: Shortcut) -> bool {
        let binding_ids = match self.by_shortcut.get(&shortcut) {
            Some(ids) => ids.clone(),
            None => return false,
        };

        // Collect matching bindings with their priorities
        let mut matches: Vec<(ShortcutId, ShortcutPriority)> = binding_ids
            .iter()
            .filter_map(|&id| {
                let binding = self.bindings.get(&id)?;
                if !binding.enabled {
                    return None;
                }
                if self.is_context_active(&binding.context) {
                    Some((id, binding.priority))
                } else {
                    None
                }
            })
            .collect();

        // Sort by priority (highest first)
        matches.sort_by(|a, b| b.1.cmp(&a.1));

        // Try handlers in priority order
        for (id, _) in matches {
            if let Some(handler) = self.handlers.get_mut(&id) {
                if handler() {
                    return true;
                }
            }
        }

        false
    }

    /// Check if a context is currently active.
    fn is_context_active(&self, context: &ShortcutContext) -> bool {
        match context {
            ShortcutContext::Global => true,
            other => self.active_contexts.contains(other),
        }
    }

    /// Get all registered shortcuts.
    pub fn shortcuts(&self) -> impl Iterator<Item = (&Shortcut, &str)> {
        self.bindings
            .values()
            .map(|b| (&b.shortcut, b.description.as_str()))
    }

    /// Get binding count.
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }

    /// Check for shortcut conflicts (same shortcut, same context).
    pub fn find_conflicts(&self) -> Vec<(Shortcut, Vec<ShortcutId>)> {
        let mut conflicts = Vec::new();

        for (shortcut, ids) in &self.by_shortcut {
            if ids.len() < 2 {
                continue;
            }

            // Group by context
            let mut by_context: HashMap<&ShortcutContext, Vec<ShortcutId>> = HashMap::new();
            for &id in ids {
                if let Some(binding) = self.bindings.get(&id) {
                    by_context.entry(&binding.context).or_default().push(id);
                }
            }

            // Find contexts with multiple bindings
            for (_, context_ids) in by_context {
                if context_ids.len() > 1 {
                    conflicts.push((*shortcut, context_ids));
                }
            }
        }

        conflicts
    }

    /// Clear all shortcuts.
    pub fn clear(&mut self) {
        self.bindings.clear();
        self.handlers.clear();
        self.by_shortcut.clear();
    }

    /// Get the description for a shortcut binding.
    pub fn description(&self, id: ShortcutId) -> Option<&str> {
        self.bindings.get(&id).map(|b| b.description.as_str())
    }
}

impl Default for ShortcutManager {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ShortcutManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShortcutManager")
            .field("binding_count", &self.bindings.len())
            .field("active_contexts", &self.active_contexts)
            .field("modifiers", &self.modifiers)
            .finish()
    }
}

/// Builder for creating shortcuts with fluent API.
#[derive(Debug, Clone)]
pub struct ShortcutBuilder {
    key: Key,
    modifiers: Modifiers,
    context: ShortcutContext,
    priority: ShortcutPriority,
    description: String,
}

impl ShortcutBuilder {
    /// Create a new builder.
    pub fn new(key: Key) -> Self {
        Self {
            key,
            modifiers: Modifiers::NONE,
            context: ShortcutContext::Global,
            priority: ShortcutPriority::Normal,
            description: String::new(),
        }
    }

    /// Add Ctrl modifier.
    pub fn ctrl(mut self) -> Self {
        self.modifiers.ctrl = true;
        self
    }

    /// Add Alt modifier.
    pub fn alt(mut self) -> Self {
        self.modifiers.alt = true;
        self
    }

    /// Add Shift modifier.
    pub fn shift(mut self) -> Self {
        self.modifiers.shift = true;
        self
    }

    /// Add Meta modifier.
    pub fn meta(mut self) -> Self {
        self.modifiers.meta = true;
        self
    }

    /// Set context.
    pub fn context(mut self, context: ShortcutContext) -> Self {
        self.context = context;
        self
    }

    /// Set context to a specific widget.
    pub fn for_widget(mut self, widget_id: WidgetId) -> Self {
        self.context = ShortcutContext::Widget(widget_id);
        self
    }

    /// Set priority.
    pub fn priority(mut self, priority: ShortcutPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set description.
    pub fn description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Build the shortcut.
    pub fn build(self) -> Shortcut {
        Shortcut::new(self.key, self.modifiers)
    }

    /// Register with a manager.
    pub fn register(self, manager: &mut ShortcutManager, handler: ShortcutHandler) -> ShortcutId {
        let shortcut = Shortcut::new(self.key, self.modifiers);
        manager.register_with_options(
            shortcut,
            handler,
            self.context,
            self.priority,
            &self.description,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    // Modifiers tests
    #[allow(clippy::assertions_on_constants)]
    #[test]
    fn test_modifiers_constants() {
        assert!(!Modifiers::NONE.any());
        assert!(Modifiers::NONE.none());

        assert!(Modifiers::CTRL.ctrl);
        assert!(!Modifiers::CTRL.alt);

        assert!(Modifiers::CTRL_SHIFT.ctrl);
        assert!(Modifiers::CTRL_SHIFT.shift);
    }

    #[test]
    fn test_modifiers_new() {
        let mods = Modifiers::new(true, true, false, false);
        assert!(mods.ctrl);
        assert!(mods.alt);
        assert!(!mods.shift);
        assert!(!mods.meta);
    }

    #[test]
    fn test_modifiers_display() {
        assert_eq!(Modifiers::NONE.display(), "");
        assert_eq!(Modifiers::CTRL.display(), "Ctrl");
        assert_eq!(Modifiers::CTRL_SHIFT.display(), "Ctrl+Shift");
    }

    // Shortcut tests
    #[test]
    fn test_shortcut_new() {
        let shortcut = Shortcut::new(Key::A, Modifiers::CTRL);
        assert_eq!(shortcut.key, Key::A);
        assert!(shortcut.modifiers.ctrl);
    }

    #[test]
    fn test_shortcut_constructors() {
        let key_only = Shortcut::key(Key::Escape);
        assert!(key_only.modifiers.none());

        let ctrl = Shortcut::ctrl(Key::S);
        assert!(ctrl.modifiers.ctrl);

        let alt = Shortcut::alt(Key::F4);
        assert!(alt.modifiers.alt);

        let shift = Shortcut::shift(Key::Tab);
        assert!(shift.modifiers.shift);

        let ctrl_shift = Shortcut::ctrl_shift(Key::Z);
        assert!(ctrl_shift.modifiers.ctrl);
        assert!(ctrl_shift.modifiers.shift);
    }

    #[test]
    fn test_shortcut_display() {
        assert_eq!(Shortcut::key(Key::A).display(), "A");
        assert_eq!(Shortcut::ctrl(Key::S).display(), "Ctrl+S");
        assert_eq!(Shortcut::ctrl_shift(Key::Z).display(), "Ctrl+Shift+Z");
    }

    #[test]
    fn test_shortcut_constants() {
        assert_eq!(Shortcut::COPY, Shortcut::ctrl(Key::C));
        assert_eq!(Shortcut::UNDO, Shortcut::ctrl(Key::Z));
        assert_eq!(Shortcut::REDO, Shortcut::ctrl_shift(Key::Z));
    }

    #[test]
    fn test_shortcut_equality() {
        let s1 = Shortcut::ctrl(Key::S);
        let s2 = Shortcut::ctrl(Key::S);
        let s3 = Shortcut::ctrl(Key::A);

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    // ShortcutId tests
    #[test]
    fn test_shortcut_id() {
        let id1 = ShortcutId::new(1);
        let id2 = ShortcutId::new(1);
        let id3 = ShortcutId::new(2);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    // ShortcutContext tests
    #[test]
    fn test_shortcut_context_default() {
        assert_eq!(ShortcutContext::default(), ShortcutContext::Global);
    }

    // ShortcutPriority tests
    #[test]
    fn test_shortcut_priority_ordering() {
        assert!(ShortcutPriority::High > ShortcutPriority::Normal);
        assert!(ShortcutPriority::Normal > ShortcutPriority::Low);
    }

    // ShortcutManager tests
    #[test]
    fn test_manager_new() {
        let manager = ShortcutManager::new();
        assert_eq!(manager.binding_count(), 0);
    }

    #[test]
    fn test_manager_register() {
        let mut manager = ShortcutManager::new();

        let id = manager.register(Shortcut::ctrl(Key::S), Box::new(|| true));
        assert_eq!(manager.binding_count(), 1);
        assert!(manager.is_enabled(id));
    }

    #[test]
    fn test_manager_unregister() {
        let mut manager = ShortcutManager::new();

        let id = manager.register(Shortcut::ctrl(Key::S), Box::new(|| true));
        assert_eq!(manager.binding_count(), 1);

        let removed = manager.unregister(id);
        assert!(removed);
        assert_eq!(manager.binding_count(), 0);
    }

    #[test]
    fn test_manager_set_enabled() {
        let mut manager = ShortcutManager::new();

        let id = manager.register(Shortcut::ctrl(Key::S), Box::new(|| true));
        assert!(manager.is_enabled(id));

        manager.set_enabled(id, false);
        assert!(!manager.is_enabled(id));

        manager.set_enabled(id, true);
        assert!(manager.is_enabled(id));
    }

    #[test]
    fn test_manager_handle_key() {
        let triggered = Arc::new(AtomicBool::new(false));
        let triggered_clone = triggered.clone();

        let mut manager = ShortcutManager::new();
        manager.register(
            Shortcut::ctrl(Key::S),
            Box::new(move || {
                triggered_clone.store(true, Ordering::SeqCst);
                true
            }),
        );

        // Without Ctrl, should not trigger
        manager.set_modifiers(Modifiers::NONE);
        let result = manager.handle_key(Key::S);
        assert!(!result);
        assert!(!triggered.load(Ordering::SeqCst));

        // With Ctrl, should trigger
        manager.set_modifiers(Modifiers::CTRL);
        let result = manager.handle_key(Key::S);
        assert!(result);
        assert!(triggered.load(Ordering::SeqCst));
    }

    #[test]
    fn test_manager_trigger() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let mut manager = ShortcutManager::new();
        manager.register(
            Shortcut::ctrl(Key::C),
            Box::new(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                true
            }),
        );

        manager.trigger(Shortcut::ctrl(Key::C));
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        manager.trigger(Shortcut::ctrl(Key::C));
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_manager_disabled_shortcut_not_triggered() {
        let triggered = Arc::new(AtomicBool::new(false));
        let triggered_clone = triggered.clone();

        let mut manager = ShortcutManager::new();
        let id = manager.register(
            Shortcut::ctrl(Key::S),
            Box::new(move || {
                triggered_clone.store(true, Ordering::SeqCst);
                true
            }),
        );

        manager.set_enabled(id, false);
        manager.set_modifiers(Modifiers::CTRL);

        let result = manager.handle_key(Key::S);
        assert!(!result);
        assert!(!triggered.load(Ordering::SeqCst));
    }

    #[test]
    fn test_manager_context() {
        let triggered = Arc::new(AtomicBool::new(false));
        let triggered_clone = triggered.clone();

        let mut manager = ShortcutManager::new();
        manager.register_with_options(
            Shortcut::ctrl(Key::S),
            Box::new(move || {
                triggered_clone.store(true, Ordering::SeqCst);
                true
            }),
            ShortcutContext::Widget(WidgetId::new(1)),
            ShortcutPriority::Normal,
            "",
        );

        // Without widget context, should not trigger
        manager.set_modifiers(Modifiers::CTRL);
        let result = manager.handle_key(Key::S);
        assert!(!result);

        // With widget context, should trigger
        manager.set_focused_widget(Some(WidgetId::new(1)));
        let result = manager.handle_key(Key::S);
        assert!(result);
        assert!(triggered.load(Ordering::SeqCst));
    }

    #[test]
    fn test_manager_priority() {
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let order1 = order.clone();
        let order2 = order.clone();

        let mut manager = ShortcutManager::new();

        manager.register_with_options(
            Shortcut::ctrl(Key::S),
            Box::new(move || {
                order1.lock().unwrap().push("low");
                false // Don't consume, let next handler run
            }),
            ShortcutContext::Global,
            ShortcutPriority::Low,
            "",
        );

        manager.register_with_options(
            Shortcut::ctrl(Key::S),
            Box::new(move || {
                order2.lock().unwrap().push("high");
                false
            }),
            ShortcutContext::Global,
            ShortcutPriority::High,
            "",
        );

        manager.trigger(Shortcut::ctrl(Key::S));

        let order_vec = order.lock().unwrap();
        assert_eq!(*order_vec, vec!["high", "low"]);
    }

    #[test]
    fn test_manager_handler_consumes() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c1 = counter.clone();
        let c2 = counter.clone();

        let mut manager = ShortcutManager::new();

        manager.register_with_options(
            Shortcut::ctrl(Key::S),
            Box::new(move || {
                c1.fetch_add(1, Ordering::SeqCst);
                true // Consume the event
            }),
            ShortcutContext::Global,
            ShortcutPriority::High,
            "",
        );

        manager.register_with_options(
            Shortcut::ctrl(Key::S),
            Box::new(move || {
                c2.fetch_add(1, Ordering::SeqCst);
                true
            }),
            ShortcutContext::Global,
            ShortcutPriority::Low,
            "",
        );

        manager.trigger(Shortcut::ctrl(Key::S));

        // Only high priority handler should have run
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_manager_push_pop_context() {
        let mut manager = ShortcutManager::new();

        manager.push_context(ShortcutContext::Custom("editor".to_string()));
        manager.push_context(ShortcutContext::Custom("modal".to_string()));

        let popped = manager.pop_context();
        assert_eq!(popped, Some(ShortcutContext::Custom("modal".to_string())));

        let popped = manager.pop_context();
        assert_eq!(popped, Some(ShortcutContext::Custom("editor".to_string())));

        // Can't pop Global context
        let popped = manager.pop_context();
        assert!(popped.is_none());
    }

    #[test]
    fn test_manager_find_conflicts() {
        let mut manager = ShortcutManager::new();

        let id1 = manager.register(Shortcut::ctrl(Key::S), Box::new(|| true));
        let id2 = manager.register(Shortcut::ctrl(Key::S), Box::new(|| true));
        manager.register(Shortcut::ctrl(Key::A), Box::new(|| true)); // No conflict

        let conflicts = manager.find_conflicts();
        assert_eq!(conflicts.len(), 1);
        assert!(conflicts[0].1.contains(&id1));
        assert!(conflicts[0].1.contains(&id2));
    }

    #[test]
    fn test_manager_shortcuts() {
        let mut manager = ShortcutManager::new();

        manager.register_with_options(
            Shortcut::ctrl(Key::S),
            Box::new(|| true),
            ShortcutContext::Global,
            ShortcutPriority::Normal,
            "Save",
        );

        manager.register_with_options(
            Shortcut::ctrl(Key::O),
            Box::new(|| true),
            ShortcutContext::Global,
            ShortcutPriority::Normal,
            "Open",
        );

        assert_eq!(manager.shortcuts().count(), 2);
    }

    #[test]
    fn test_manager_clear() {
        let mut manager = ShortcutManager::new();
        manager.register(Shortcut::ctrl(Key::S), Box::new(|| true));
        manager.register(Shortcut::ctrl(Key::O), Box::new(|| true));

        manager.clear();
        assert_eq!(manager.binding_count(), 0);
    }

    #[test]
    fn test_manager_description() {
        let mut manager = ShortcutManager::new();

        let id = manager.register_with_options(
            Shortcut::ctrl(Key::S),
            Box::new(|| true),
            ShortcutContext::Global,
            ShortcutPriority::Normal,
            "Save document",
        );

        assert_eq!(manager.description(id), Some("Save document"));
    }

    // ShortcutBuilder tests
    #[test]
    fn test_builder() {
        let shortcut = ShortcutBuilder::new(Key::S).ctrl().shift().build();

        assert_eq!(shortcut.key, Key::S);
        assert!(shortcut.modifiers.ctrl);
        assert!(shortcut.modifiers.shift);
        assert!(!shortcut.modifiers.alt);
    }

    #[test]
    fn test_builder_register() {
        let mut manager = ShortcutManager::new();

        let id = ShortcutBuilder::new(Key::S)
            .ctrl()
            .description("Save")
            .register(&mut manager, Box::new(|| true));

        assert!(manager.is_enabled(id));
        assert_eq!(manager.description(id), Some("Save"));
    }

    #[test]
    fn test_builder_for_widget() {
        let builder = ShortcutBuilder::new(Key::Enter).for_widget(WidgetId::new(42));

        assert_eq!(builder.context, ShortcutContext::Widget(WidgetId::new(42)));
    }
}
