#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! Widget lifecycle hooks for mount, update, and unmount callbacks.
//!
//! This module provides a system for managing widget lifecycle events,
//! similar to React's useEffect or Vue's lifecycle hooks.

use crate::widget::WidgetId;
use std::collections::HashMap;

/// Lifecycle phase for widgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LifecyclePhase {
    /// Widget is being created/mounted.
    Mount,
    /// Widget is being updated (props/state changed).
    Update,
    /// Widget is being removed/unmounted.
    Unmount,
    /// Before the paint phase.
    BeforePaint,
    /// After the paint phase.
    AfterPaint,
    /// Widget gained focus.
    Focus,
    /// Widget lost focus.
    Blur,
    /// Widget became visible.
    Visible,
    /// Widget became hidden.
    Hidden,
}

/// A lifecycle callback that can be registered.
pub type LifecycleCallback = Box<dyn FnMut(LifecycleEvent) + Send>;

/// Event passed to lifecycle callbacks.
#[derive(Debug, Clone)]
pub struct LifecycleEvent {
    /// Widget ID.
    pub widget_id: WidgetId,
    /// Phase of the lifecycle.
    pub phase: LifecyclePhase,
    /// Timestamp (frame number or monotonic counter).
    pub timestamp: u64,
}

impl LifecycleEvent {
    /// Create a new lifecycle event.
    pub fn new(widget_id: WidgetId, phase: LifecyclePhase, timestamp: u64) -> Self {
        Self {
            widget_id,
            phase,
            timestamp,
        }
    }
}

/// Unique ID for a lifecycle hook registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HookId(pub u64);

impl HookId {
    /// Create a new hook ID.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Registration for a lifecycle hook.
#[derive(Debug)]
struct HookRegistration {
    #[allow(dead_code)]
    id: HookId,
    widget_id: WidgetId,
    phases: Vec<LifecyclePhase>,
}

/// Manager for widget lifecycle hooks.
pub struct LifecycleManager {
    /// Next hook ID.
    next_id: u64,
    /// Registered hooks (by hook ID).
    hooks: HashMap<HookId, HookRegistration>,
    /// Callbacks (by hook ID).
    callbacks: HashMap<HookId, LifecycleCallback>,
    /// Index of hooks by widget ID.
    by_widget: HashMap<WidgetId, Vec<HookId>>,
    /// Index of hooks by phase.
    by_phase: HashMap<LifecyclePhase, Vec<HookId>>,
    /// Current frame/timestamp.
    timestamp: u64,
    /// Pending events to dispatch.
    pending_events: Vec<LifecycleEvent>,
}

impl LifecycleManager {
    /// Create a new lifecycle manager.
    pub fn new() -> Self {
        Self {
            next_id: 0,
            hooks: HashMap::new(),
            callbacks: HashMap::new(),
            by_widget: HashMap::new(),
            by_phase: HashMap::new(),
            timestamp: 0,
            pending_events: Vec::new(),
        }
    }

    /// Register a lifecycle hook.
    ///
    /// Returns a hook ID that can be used to unregister the hook.
    pub fn register(
        &mut self,
        widget_id: WidgetId,
        phases: Vec<LifecyclePhase>,
        callback: LifecycleCallback,
    ) -> HookId {
        let id = HookId::new(self.next_id);
        self.next_id += 1;

        let registration = HookRegistration {
            id,
            widget_id,
            phases: phases.clone(),
        };

        self.hooks.insert(id, registration);
        self.callbacks.insert(id, callback);

        // Index by widget
        self.by_widget.entry(widget_id).or_default().push(id);

        // Index by phase
        for phase in phases {
            self.by_phase.entry(phase).or_default().push(id);
        }

        id
    }

    /// Register a mount hook.
    pub fn on_mount(&mut self, widget_id: WidgetId, callback: LifecycleCallback) -> HookId {
        self.register(widget_id, vec![LifecyclePhase::Mount], callback)
    }

    /// Register an unmount hook.
    pub fn on_unmount(&mut self, widget_id: WidgetId, callback: LifecycleCallback) -> HookId {
        self.register(widget_id, vec![LifecyclePhase::Unmount], callback)
    }

    /// Register an update hook.
    pub fn on_update(&mut self, widget_id: WidgetId, callback: LifecycleCallback) -> HookId {
        self.register(widget_id, vec![LifecyclePhase::Update], callback)
    }

    /// Register a focus hook.
    pub fn on_focus(&mut self, widget_id: WidgetId, callback: LifecycleCallback) -> HookId {
        self.register(widget_id, vec![LifecyclePhase::Focus], callback)
    }

    /// Register a blur hook.
    pub fn on_blur(&mut self, widget_id: WidgetId, callback: LifecycleCallback) -> HookId {
        self.register(widget_id, vec![LifecyclePhase::Blur], callback)
    }

    /// Unregister a hook.
    pub fn unregister(&mut self, hook_id: HookId) -> bool {
        if let Some(registration) = self.hooks.remove(&hook_id) {
            self.callbacks.remove(&hook_id);

            // Remove from widget index
            if let Some(hooks) = self.by_widget.get_mut(&registration.widget_id) {
                hooks.retain(|&id| id != hook_id);
            }

            // Remove from phase index
            for phase in &registration.phases {
                if let Some(hooks) = self.by_phase.get_mut(phase) {
                    hooks.retain(|&id| id != hook_id);
                }
            }

            true
        } else {
            false
        }
    }

    /// Unregister all hooks for a widget.
    pub fn unregister_widget(&mut self, widget_id: WidgetId) {
        if let Some(hook_ids) = self.by_widget.remove(&widget_id) {
            for hook_id in hook_ids {
                if let Some(registration) = self.hooks.remove(&hook_id) {
                    self.callbacks.remove(&hook_id);

                    for phase in &registration.phases {
                        if let Some(hooks) = self.by_phase.get_mut(phase) {
                            hooks.retain(|&id| id != hook_id);
                        }
                    }
                }
            }
        }
    }

    /// Emit a lifecycle event immediately.
    pub fn emit(&mut self, widget_id: WidgetId, phase: LifecyclePhase) {
        let event = LifecycleEvent::new(widget_id, phase, self.timestamp);

        // Get hooks for this widget and phase
        let widget_hooks = self.by_widget.get(&widget_id).cloned().unwrap_or_default();
        let phase_hooks = self.by_phase.get(&phase).cloned().unwrap_or_default();

        // Find intersection (hooks registered for both this widget and phase)
        for hook_id in widget_hooks {
            if phase_hooks.contains(&hook_id) {
                if let Some(callback) = self.callbacks.get_mut(&hook_id) {
                    callback(event.clone());
                }
            }
        }
    }

    /// Queue a lifecycle event for later dispatch.
    pub fn queue(&mut self, widget_id: WidgetId, phase: LifecyclePhase) {
        let event = LifecycleEvent::new(widget_id, phase, self.timestamp);
        self.pending_events.push(event);
    }

    /// Dispatch all pending events.
    pub fn flush(&mut self) {
        let events: Vec<LifecycleEvent> = self.pending_events.drain(..).collect();

        for event in events {
            let widget_hooks = self
                .by_widget
                .get(&event.widget_id)
                .cloned()
                .unwrap_or_default();
            let phase_hooks = self.by_phase.get(&event.phase).cloned().unwrap_or_default();

            for hook_id in widget_hooks {
                if phase_hooks.contains(&hook_id) {
                    if let Some(callback) = self.callbacks.get_mut(&hook_id) {
                        callback(event.clone());
                    }
                }
            }
        }
    }

    /// Get the number of pending events.
    pub fn pending_count(&self) -> usize {
        self.pending_events.len()
    }

    /// Advance the timestamp.
    pub fn tick(&mut self) {
        self.timestamp += 1;
    }

    /// Get the current timestamp.
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Get the number of registered hooks.
    pub fn hook_count(&self) -> usize {
        self.hooks.len()
    }

    /// Check if a widget has any hooks.
    pub fn has_hooks(&self, widget_id: WidgetId) -> bool {
        self.by_widget
            .get(&widget_id)
            .is_some_and(|h| !h.is_empty())
    }

    /// Clear all hooks and events.
    pub fn clear(&mut self) {
        self.hooks.clear();
        self.callbacks.clear();
        self.by_widget.clear();
        self.by_phase.clear();
        self.pending_events.clear();
    }
}

impl Default for LifecycleManager {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for LifecycleManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LifecycleManager")
            .field("next_id", &self.next_id)
            .field("hook_count", &self.hooks.len())
            .field("timestamp", &self.timestamp)
            .field("pending_count", &self.pending_events.len())
            .finish()
    }
}

/// Effect hook that runs a callback and optionally cleans up.
pub struct Effect {
    /// Effect function that returns an optional cleanup function.
    effect: Option<Box<dyn FnOnce() -> Option<Box<dyn FnOnce() + Send>> + Send>>,
    /// Cleanup function from the last run.
    cleanup: Option<Box<dyn FnOnce() + Send>>,
    /// Dependencies for determining when to re-run.
    deps: Vec<u64>,
}

impl Effect {
    /// Create a new effect.
    pub fn new<F>(effect: F) -> Self
    where
        F: FnOnce() -> Option<Box<dyn FnOnce() + Send>> + Send + 'static,
    {
        Self {
            effect: Some(Box::new(effect)),
            cleanup: None,
            deps: Vec::new(),
        }
    }

    /// Create an effect with dependencies.
    pub fn with_deps<F>(effect: F, deps: Vec<u64>) -> Self
    where
        F: FnOnce() -> Option<Box<dyn FnOnce() + Send>> + Send + 'static,
    {
        Self {
            effect: Some(Box::new(effect)),
            cleanup: None,
            deps,
        }
    }

    /// Check if dependencies changed.
    pub fn deps_changed(&self, new_deps: &[u64]) -> bool {
        if self.deps.len() != new_deps.len() {
            return true;
        }
        self.deps.iter().zip(new_deps).any(|(a, b)| a != b)
    }

    /// Run the effect if dependencies changed.
    pub fn run(&mut self, new_deps: Option<&[u64]>) -> bool {
        // Check if we should run based on deps
        let should_run = match new_deps {
            Some(deps) if !self.deps_changed(deps) => false,
            _ => true,
        };

        if !should_run {
            return false;
        }

        // Run cleanup from previous effect
        if let Some(cleanup) = self.cleanup.take() {
            cleanup();
        }

        // Run the effect
        if let Some(effect) = self.effect.take() {
            self.cleanup = effect();
        }

        // Update deps
        if let Some(deps) = new_deps {
            self.deps = deps.to_vec();
        }

        true
    }

    /// Run cleanup without running the effect.
    pub fn cleanup(&mut self) {
        if let Some(cleanup) = self.cleanup.take() {
            cleanup();
        }
    }

    /// Check if the effect has a pending cleanup.
    pub fn has_cleanup(&self) -> bool {
        self.cleanup.is_some()
    }
}

impl std::fmt::Debug for Effect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Effect")
            .field("has_effect", &self.effect.is_some())
            .field("has_cleanup", &self.cleanup.is_some())
            .field("deps", &self.deps)
            .finish()
    }
}

/// Manager for effects with automatic cleanup.
#[derive(Debug, Default)]
pub struct EffectManager {
    /// Effects by widget ID.
    effects: HashMap<WidgetId, Vec<Effect>>,
}

impl EffectManager {
    /// Create a new effect manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an effect for a widget.
    pub fn add(&mut self, widget_id: WidgetId, effect: Effect) {
        self.effects.entry(widget_id).or_default().push(effect);
    }

    /// Run all effects for a widget.
    pub fn run_effects(&mut self, widget_id: WidgetId, deps: Option<&[u64]>) {
        if let Some(effects) = self.effects.get_mut(&widget_id) {
            for effect in effects {
                effect.run(deps);
            }
        }
    }

    /// Clean up effects for a widget (e.g., on unmount).
    pub fn cleanup_widget(&mut self, widget_id: WidgetId) {
        if let Some(effects) = self.effects.get_mut(&widget_id) {
            for effect in effects {
                effect.cleanup();
            }
        }
        self.effects.remove(&widget_id);
    }

    /// Get the number of widgets with effects.
    pub fn widget_count(&self) -> usize {
        self.effects.len()
    }

    /// Get the total number of effects.
    pub fn effect_count(&self) -> usize {
        self.effects.values().map(std::vec::Vec::len).sum()
    }

    /// Clear all effects.
    pub fn clear(&mut self) {
        for effects in self.effects.values_mut() {
            for effect in effects {
                effect.cleanup();
            }
        }
        self.effects.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // LifecyclePhase tests
    #[test]
    fn test_lifecycle_phase_equality() {
        assert_eq!(LifecyclePhase::Mount, LifecyclePhase::Mount);
        assert_ne!(LifecyclePhase::Mount, LifecyclePhase::Unmount);
    }

    // LifecycleEvent tests
    #[test]
    fn test_lifecycle_event_new() {
        let event = LifecycleEvent::new(WidgetId::new(1), LifecyclePhase::Mount, 42);
        assert_eq!(event.widget_id, WidgetId::new(1));
        assert_eq!(event.phase, LifecyclePhase::Mount);
        assert_eq!(event.timestamp, 42);
    }

    // HookId tests
    #[test]
    fn test_hook_id() {
        let id1 = HookId::new(1);
        let id2 = HookId::new(1);
        let id3 = HookId::new(2);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    // LifecycleManager tests
    #[test]
    fn test_manager_new() {
        let manager = LifecycleManager::new();
        assert_eq!(manager.hook_count(), 0);
        assert_eq!(manager.timestamp(), 0);
    }

    #[test]
    fn test_manager_register() {
        let mut manager = LifecycleManager::new();
        let widget_id = WidgetId::new(1);

        let hook_id = manager.register(widget_id, vec![LifecyclePhase::Mount], Box::new(|_| {}));

        assert_eq!(manager.hook_count(), 1);
        assert!(manager.has_hooks(widget_id));
        assert!(!manager.has_hooks(WidgetId::new(999)));
        assert_eq!(hook_id.0, 0);
    }

    #[test]
    fn test_manager_on_mount() {
        let mut manager = LifecycleManager::new();
        let widget_id = WidgetId::new(1);

        let _hook_id = manager.on_mount(widget_id, Box::new(|_| {}));
        assert_eq!(manager.hook_count(), 1);
    }

    #[test]
    fn test_manager_on_unmount() {
        let mut manager = LifecycleManager::new();
        let widget_id = WidgetId::new(1);

        let _hook_id = manager.on_unmount(widget_id, Box::new(|_| {}));
        assert_eq!(manager.hook_count(), 1);
    }

    #[test]
    fn test_manager_emit() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let mut manager = LifecycleManager::new();
        let widget_id = WidgetId::new(1);

        manager.on_mount(
            widget_id,
            Box::new(move |_| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }),
        );

        manager.emit(widget_id, LifecyclePhase::Mount);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Emit again
        manager.emit(widget_id, LifecyclePhase::Mount);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_manager_emit_wrong_phase() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let mut manager = LifecycleManager::new();
        let widget_id = WidgetId::new(1);

        manager.on_mount(
            widget_id,
            Box::new(move |_| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }),
        );

        // Emit unmount instead of mount
        manager.emit(widget_id, LifecyclePhase::Unmount);
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_manager_queue_and_flush() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let mut manager = LifecycleManager::new();
        let widget_id = WidgetId::new(1);

        manager.on_mount(
            widget_id,
            Box::new(move |_| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }),
        );

        manager.queue(widget_id, LifecyclePhase::Mount);
        manager.queue(widget_id, LifecyclePhase::Mount);
        assert_eq!(manager.pending_count(), 2);
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        manager.flush();
        assert_eq!(manager.pending_count(), 0);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_manager_unregister() {
        let mut manager = LifecycleManager::new();
        let widget_id = WidgetId::new(1);

        let hook_id = manager.on_mount(widget_id, Box::new(|_| {}));
        assert_eq!(manager.hook_count(), 1);

        let removed = manager.unregister(hook_id);
        assert!(removed);
        assert_eq!(manager.hook_count(), 0);
        assert!(!manager.has_hooks(widget_id));
    }

    #[test]
    fn test_manager_unregister_widget() {
        let mut manager = LifecycleManager::new();
        let widget_id = WidgetId::new(1);

        manager.on_mount(widget_id, Box::new(|_| {}));
        manager.on_unmount(widget_id, Box::new(|_| {}));
        manager.on_update(widget_id, Box::new(|_| {}));
        assert_eq!(manager.hook_count(), 3);

        manager.unregister_widget(widget_id);
        assert_eq!(manager.hook_count(), 0);
    }

    #[test]
    fn test_manager_tick() {
        let mut manager = LifecycleManager::new();
        assert_eq!(manager.timestamp(), 0);

        manager.tick();
        assert_eq!(manager.timestamp(), 1);

        manager.tick();
        manager.tick();
        assert_eq!(manager.timestamp(), 3);
    }

    #[test]
    fn test_manager_clear() {
        let mut manager = LifecycleManager::new();
        let widget_id = WidgetId::new(1);

        manager.on_mount(widget_id, Box::new(|_| {}));
        manager.queue(widget_id, LifecyclePhase::Mount);

        manager.clear();
        assert_eq!(manager.hook_count(), 0);
        assert_eq!(manager.pending_count(), 0);
    }

    #[test]
    fn test_manager_multiple_widgets() {
        let counter1 = Arc::new(AtomicUsize::new(0));
        let counter2 = Arc::new(AtomicUsize::new(0));
        let c1 = counter1.clone();
        let c2 = counter2.clone();

        let mut manager = LifecycleManager::new();

        manager.on_mount(
            WidgetId::new(1),
            Box::new(move |_| {
                c1.fetch_add(1, Ordering::SeqCst);
            }),
        );
        manager.on_mount(
            WidgetId::new(2),
            Box::new(move |_| {
                c2.fetch_add(1, Ordering::SeqCst);
            }),
        );

        manager.emit(WidgetId::new(1), LifecyclePhase::Mount);
        assert_eq!(counter1.load(Ordering::SeqCst), 1);
        assert_eq!(counter2.load(Ordering::SeqCst), 0);

        manager.emit(WidgetId::new(2), LifecyclePhase::Mount);
        assert_eq!(counter1.load(Ordering::SeqCst), 1);
        assert_eq!(counter2.load(Ordering::SeqCst), 1);
    }

    // Effect tests
    #[test]
    fn test_effect_new() {
        let effect = Effect::new(|| None);
        assert!(!effect.has_cleanup());
    }

    #[test]
    fn test_effect_with_deps() {
        let effect = Effect::with_deps(|| None, vec![1, 2, 3]);
        assert_eq!(effect.deps, vec![1, 2, 3]);
    }

    #[test]
    fn test_effect_deps_changed() {
        let effect = Effect::with_deps(|| None, vec![1, 2, 3]);

        assert!(!effect.deps_changed(&[1, 2, 3]));
        assert!(effect.deps_changed(&[1, 2, 4]));
        assert!(effect.deps_changed(&[1, 2]));
        assert!(effect.deps_changed(&[1, 2, 3, 4]));
    }

    #[test]
    fn test_effect_run() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();

        let mut effect = Effect::new(move || {
            c.fetch_add(1, Ordering::SeqCst);
            None
        });

        effect.run(None);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Effect can only run once (it's moved out)
        effect.run(None);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_effect_cleanup() {
        let cleanup_counter = Arc::new(AtomicUsize::new(0));
        let cc = cleanup_counter.clone();

        let mut effect = Effect::new(move || {
            let cc = cc;
            Some(Box::new(move || {
                cc.fetch_add(1, Ordering::SeqCst);
            }) as Box<dyn FnOnce() + Send>)
        });

        effect.run(None);
        assert!(effect.has_cleanup());
        assert_eq!(cleanup_counter.load(Ordering::SeqCst), 0);

        effect.cleanup();
        assert!(!effect.has_cleanup());
        assert_eq!(cleanup_counter.load(Ordering::SeqCst), 1);
    }

    // EffectManager tests
    #[test]
    fn test_effect_manager_new() {
        let manager = EffectManager::new();
        assert_eq!(manager.widget_count(), 0);
        assert_eq!(manager.effect_count(), 0);
    }

    #[test]
    fn test_effect_manager_add() {
        let mut manager = EffectManager::new();
        let widget_id = WidgetId::new(1);

        manager.add(widget_id, Effect::new(|| None));
        manager.add(widget_id, Effect::new(|| None));

        assert_eq!(manager.widget_count(), 1);
        assert_eq!(manager.effect_count(), 2);
    }

    #[test]
    fn test_effect_manager_run_effects() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();

        let mut manager = EffectManager::new();
        let widget_id = WidgetId::new(1);

        manager.add(
            widget_id,
            Effect::new(move || {
                c.fetch_add(1, Ordering::SeqCst);
                None
            }),
        );

        manager.run_effects(widget_id, None);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_effect_manager_cleanup_widget() {
        let cleanup_counter = Arc::new(AtomicUsize::new(0));
        let cc = cleanup_counter.clone();

        let mut manager = EffectManager::new();
        let widget_id = WidgetId::new(1);

        manager.add(
            widget_id,
            Effect::new(move || {
                let cc = cc;
                Some(Box::new(move || {
                    cc.fetch_add(1, Ordering::SeqCst);
                }) as Box<dyn FnOnce() + Send>)
            }),
        );

        manager.run_effects(widget_id, None);
        assert_eq!(manager.effect_count(), 1);

        manager.cleanup_widget(widget_id);
        assert_eq!(manager.effect_count(), 0);
        assert_eq!(cleanup_counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_effect_manager_clear() {
        let mut manager = EffectManager::new();

        manager.add(WidgetId::new(1), Effect::new(|| None));
        manager.add(WidgetId::new(2), Effect::new(|| None));

        manager.clear();
        assert_eq!(manager.widget_count(), 0);
        assert_eq!(manager.effect_count(), 0);
    }
}
