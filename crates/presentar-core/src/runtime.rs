//! Command runtime for executing side effects.
//!
//! This module provides the infrastructure to execute `Command` values
//! produced by state updates.

use crate::state::Command;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Router trait for navigation commands.
pub trait Router: Send + Sync {
    /// Navigate to a route.
    fn navigate(&self, route: &str);

    /// Get the current route.
    fn current_route(&self) -> String;
}

/// Storage trait for state persistence commands.
pub trait Storage: Send + Sync {
    /// Save data to storage.
    fn save(&self, key: &str, data: &[u8]);

    /// Load data from storage.
    fn load(&self, key: &str) -> Option<Vec<u8>>;

    /// Remove data from storage.
    fn remove(&self, key: &str);

    /// Check if a key exists.
    fn contains(&self, key: &str) -> bool;
}

/// In-memory storage for testing.
#[derive(Debug, Default)]
pub struct MemoryStorage {
    data: Mutex<HashMap<String, Vec<u8>>>,
}

impl MemoryStorage {
    /// Create a new empty memory storage.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the number of stored items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data
            .lock()
            .expect("MemoryStorage mutex poisoned")
            .len()
    }

    /// Check if storage is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data
            .lock()
            .expect("MemoryStorage mutex poisoned")
            .is_empty()
    }

    /// Clear all stored data.
    pub fn clear(&self) {
        self.data
            .lock()
            .expect("MemoryStorage mutex poisoned")
            .clear();
    }
}

impl Storage for MemoryStorage {
    fn save(&self, key: &str, data: &[u8]) {
        self.data
            .lock()
            .expect("MemoryStorage mutex poisoned")
            .insert(key.to_string(), data.to_vec());
    }

    fn load(&self, key: &str) -> Option<Vec<u8>> {
        self.data
            .lock()
            .expect("MemoryStorage mutex poisoned")
            .get(key)
            .cloned()
    }

    fn remove(&self, key: &str) {
        self.data
            .lock()
            .expect("MemoryStorage mutex poisoned")
            .remove(key);
    }

    fn contains(&self, key: &str) -> bool {
        self.data
            .lock()
            .expect("MemoryStorage mutex poisoned")
            .contains_key(key)
    }
}

/// In-memory router for testing.
#[derive(Debug)]
pub struct MemoryRouter {
    route: Mutex<String>,
    history: Mutex<Vec<String>>,
}

impl Default for MemoryRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryRouter {
    /// Create a new memory router.
    #[must_use]
    pub fn new() -> Self {
        Self {
            route: Mutex::new("/".to_string()),
            history: Mutex::new(vec!["/".to_string()]),
        }
    }

    /// Get navigation history.
    #[must_use]
    pub fn history(&self) -> Vec<String> {
        self.history
            .lock()
            .expect("MemoryRouter mutex poisoned")
            .clone()
    }

    /// Get history length.
    #[must_use]
    pub fn history_len(&self) -> usize {
        self.history
            .lock()
            .expect("MemoryRouter mutex poisoned")
            .len()
    }
}

impl Router for MemoryRouter {
    fn navigate(&self, route: &str) {
        let mut current = self.route.lock().expect("MemoryRouter mutex poisoned");
        *current = route.to_string();
        self.history
            .lock()
            .expect("MemoryRouter mutex poisoned")
            .push(route.to_string());
    }

    fn current_route(&self) -> String {
        self.route
            .lock()
            .expect("MemoryRouter mutex poisoned")
            .clone()
    }
}

/// Result of command execution.
#[derive(Debug)]
pub enum ExecutionResult<M> {
    /// No result (`Command::None` or non-message-producing commands)
    None,
    /// A single message was produced
    Message(M),
    /// Multiple messages were produced
    Messages(Vec<M>),
    /// Command is pending (async)
    Pending,
}

impl<M> ExecutionResult<M> {
    /// Check if the result has no messages.
    #[must_use]
    pub const fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Check if there are messages.
    #[must_use]
    pub const fn has_messages(&self) -> bool {
        matches!(self, Self::Message(_) | Self::Messages(_))
    }

    /// Get messages as a vector.
    pub fn into_messages(self) -> Vec<M> {
        match self {
            Self::None | Self::Pending => vec![],
            Self::Message(m) => vec![m],
            Self::Messages(ms) => ms,
        }
    }
}

/// Command executor configuration.
pub struct ExecutorConfig<R, S> {
    /// Router for navigation commands
    pub router: Arc<R>,
    /// Storage for persistence commands
    pub storage: Arc<S>,
}

impl<R: Router, S: Storage> ExecutorConfig<R, S> {
    /// Create a new executor config.
    pub fn new(router: R, storage: S) -> Self {
        Self {
            router: Arc::new(router),
            storage: Arc::new(storage),
        }
    }
}

/// Command executor for synchronous commands.
///
/// Note: Task commands require async execution and return `ExecutionResult::Pending`.
pub struct CommandExecutor<R, S> {
    config: ExecutorConfig<R, S>,
}

impl<R: Router, S: Storage> CommandExecutor<R, S> {
    /// Create a new command executor.
    pub const fn new(config: ExecutorConfig<R, S>) -> Self {
        Self { config }
    }

    /// Execute a command synchronously.
    ///
    /// For async Task commands, this returns `ExecutionResult::Pending`.
    /// Use `execute_blocking` to block on async tasks.
    pub fn execute<M: Send>(&self, command: Command<M>) -> ExecutionResult<M> {
        match command {
            Command::None => ExecutionResult::None,
            Command::Batch(commands) => {
                let mut messages = Vec::new();
                for cmd in commands {
                    match self.execute(cmd) {
                        ExecutionResult::None | ExecutionResult::Pending => {}
                        ExecutionResult::Message(m) => messages.push(m),
                        ExecutionResult::Messages(ms) => messages.extend(ms),
                    }
                }
                if messages.is_empty() {
                    ExecutionResult::None
                } else {
                    ExecutionResult::Messages(messages)
                }
            }
            Command::Task(_) => {
                // Async tasks can't be executed synchronously
                ExecutionResult::Pending
            }
            Command::Navigate { route } => {
                self.config.router.navigate(&route);
                ExecutionResult::None
            }
            Command::SaveState { key } => {
                // SaveState requires the actual state to be passed
                // This is a limitation - we'd need state access
                // For now, just record that we tried to save
                // In practice, the runtime would have state access
                let _ = key;
                ExecutionResult::None
            }
            Command::LoadState { key, on_load } => {
                let data = self.config.storage.load(&key);
                let message = on_load(data);
                ExecutionResult::Message(message)
            }
        }
    }

    /// Get the router.
    pub fn router(&self) -> &R {
        &self.config.router
    }

    /// Get the storage.
    pub fn storage(&self) -> &S {
        &self.config.storage
    }
}

/// Create a default executor with memory-based backends.
#[must_use]
pub fn default_executor() -> CommandExecutor<MemoryRouter, MemoryStorage> {
    CommandExecutor::new(ExecutorConfig::new(
        MemoryRouter::new(),
        MemoryStorage::new(),
    ))
}

// =============================================================================
// Focus Management
// =============================================================================

/// Focus direction for keyboard navigation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusDirection {
    /// Move focus forward (Tab)
    Forward,
    /// Move focus backward (Shift+Tab)
    Backward,
    /// Move focus up (Arrow Up)
    Up,
    /// Move focus down (Arrow Down)
    Down,
    /// Move focus left (Arrow Left)
    Left,
    /// Move focus right (Arrow Right)
    Right,
}

/// Manages keyboard focus for widgets.
#[derive(Debug, Default)]
pub struct FocusManager {
    /// Currently focused widget ID
    focused: Option<u64>,
    /// Focus ring (ordered list of focusable widget IDs)
    focus_ring: Vec<u64>,
    /// Focus trap stack (for modals/dialogs)
    traps: Vec<FocusTrap>,
}

/// A focus trap that restricts focus to a subset of widgets.
#[derive(Debug)]
pub struct FocusTrap {
    /// Widget IDs in this trap
    pub widget_ids: Vec<u64>,
    /// Initial focused widget when trap was created
    pub initial_focus: Option<u64>,
}

impl FocusManager {
    /// Create a new focus manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the focus ring (ordered list of focusable widgets).
    pub fn set_focus_ring(&mut self, widget_ids: Vec<u64>) {
        self.focus_ring = widget_ids;
    }

    /// Get the currently focused widget ID.
    #[must_use]
    pub const fn focused(&self) -> Option<u64> {
        self.focused
    }

    /// Set focus to a specific widget.
    pub fn focus(&mut self, widget_id: u64) -> bool {
        let available = self.available_focus_ring();
        if available.contains(&widget_id) {
            self.focused = Some(widget_id);
            true
        } else {
            false
        }
    }

    /// Clear focus.
    pub fn blur(&mut self) {
        self.focused = None;
    }

    /// Move focus in a direction.
    pub fn move_focus(&mut self, direction: FocusDirection) -> Option<u64> {
        let ring = self.available_focus_ring();
        if ring.is_empty() {
            return None;
        }

        let current_idx = self
            .focused
            .and_then(|f| ring.iter().position(|&id| id == f));

        let next_idx = match direction {
            FocusDirection::Forward | FocusDirection::Down | FocusDirection::Right => {
                match current_idx {
                    Some(idx) => (idx + 1) % ring.len(),
                    None => 0,
                }
            }
            FocusDirection::Backward | FocusDirection::Up | FocusDirection::Left => {
                match current_idx {
                    Some(0) | None => ring.len() - 1,
                    Some(idx) => idx - 1,
                }
            }
        };

        let next_id = ring[next_idx];
        self.focused = Some(next_id);
        Some(next_id)
    }

    /// Push a focus trap (for modals/dialogs).
    pub fn push_trap(&mut self, widget_ids: Vec<u64>) {
        let initial = self.focused;
        self.traps.push(FocusTrap {
            widget_ids,
            initial_focus: initial,
        });
        // Focus first item in trap
        if let Some(first) = self.available_focus_ring().first().copied() {
            self.focused = Some(first);
        }
    }

    /// Pop the current focus trap.
    pub fn pop_trap(&mut self) -> Option<FocusTrap> {
        let trap = self.traps.pop();
        // Restore previous focus
        if let Some(ref t) = trap {
            self.focused = t.initial_focus;
        }
        trap
    }

    /// Check if focus is currently trapped.
    #[must_use]
    pub fn is_trapped(&self) -> bool {
        !self.traps.is_empty()
    }

    /// Get the available focus ring (respecting traps).
    fn available_focus_ring(&self) -> Vec<u64> {
        if let Some(trap) = self.traps.last() {
            trap.widget_ids.clone()
        } else {
            self.focus_ring.clone()
        }
    }

    /// Check if a widget is focusable.
    #[must_use]
    pub fn is_focusable(&self, widget_id: u64) -> bool {
        self.available_focus_ring().contains(&widget_id)
    }
}

// =============================================================================
// Animation & Timer System
// =============================================================================

/// Easing functions for smooth animations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EasingFunction {
    /// Linear interpolation (no easing)
    #[default]
    Linear,
    /// Quadratic ease in
    EaseInQuad,
    /// Quadratic ease out
    EaseOutQuad,
    /// Quadratic ease in-out
    EaseInOutQuad,
    /// Cubic ease in
    EaseInCubic,
    /// Cubic ease out
    EaseOutCubic,
    /// Cubic ease in-out
    EaseInOutCubic,
    /// Elastic ease out (spring-like)
    EaseOutElastic,
    /// Bounce ease out
    EaseOutBounce,
}

impl EasingFunction {
    /// Apply the easing function to a normalized time value (0.0 to 1.0).
    #[must_use]
    #[allow(clippy::suboptimal_flops)]
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseInQuad => t * t,
            Self::EaseOutQuad => 1.0 - (1.0 - t) * (1.0 - t),
            Self::EaseInOutQuad => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
                }
            }
            Self::EaseInCubic => t * t * t,
            Self::EaseOutCubic => 1.0 - (1.0 - t).powi(3),
            Self::EaseInOutCubic => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
                }
            }
            Self::EaseOutElastic => {
                if t == 0.0 || t == 1.0 {
                    t
                } else {
                    let c4 = (2.0 * std::f32::consts::PI) / 3.0;
                    2.0_f32.powf(-10.0 * t) * ((t * 10.0 - 0.75) * c4).sin() + 1.0
                }
            }
            Self::EaseOutBounce => {
                let n1 = 7.5625;
                let d1 = 2.75;
                if t < 1.0 / d1 {
                    n1 * t * t
                } else if t < 2.0 / d1 {
                    let t = t - 1.5 / d1;
                    n1 * t * t + 0.75
                } else if t < 2.5 / d1 {
                    let t = t - 2.25 / d1;
                    n1 * t * t + 0.9375
                } else {
                    let t = t - 2.625 / d1;
                    n1 * t * t + 0.984_375
                }
            }
        }
    }
}

/// A tween that interpolates between two values over time.
#[derive(Debug, Clone)]
pub struct Tween<T> {
    /// Starting value
    pub from: T,
    /// Ending value
    pub to: T,
    /// Duration in milliseconds
    pub duration_ms: u32,
    /// Easing function
    pub easing: EasingFunction,
    /// Current elapsed time in milliseconds
    elapsed_ms: u32,
}

impl<T: Clone> Tween<T> {
    /// Create a new tween.
    pub fn new(from: T, to: T, duration_ms: u32) -> Self {
        Self {
            from,
            to,
            duration_ms,
            easing: EasingFunction::default(),
            elapsed_ms: 0,
        }
    }

    /// Set the easing function.
    #[must_use]
    pub const fn with_easing(mut self, easing: EasingFunction) -> Self {
        self.easing = easing;
        self
    }

    /// Get the normalized progress (0.0 to 1.0).
    #[must_use]
    pub fn progress(&self) -> f32 {
        if self.duration_ms == 0 {
            1.0
        } else {
            (self.elapsed_ms as f32 / self.duration_ms as f32).min(1.0)
        }
    }

    /// Get the eased progress value.
    #[must_use]
    pub fn eased_progress(&self) -> f32 {
        self.easing.apply(self.progress())
    }

    /// Check if the tween is complete.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.elapsed_ms >= self.duration_ms
    }

    /// Advance the tween by delta milliseconds.
    pub fn advance(&mut self, delta_ms: u32) {
        self.elapsed_ms = self
            .elapsed_ms
            .saturating_add(delta_ms)
            .min(self.duration_ms);
    }

    /// Reset the tween to the beginning.
    pub fn reset(&mut self) {
        self.elapsed_ms = 0;
    }
}

impl Tween<f32> {
    /// Get the current interpolated value.
    #[must_use]
    #[allow(clippy::suboptimal_flops)]
    pub fn value(&self) -> f32 {
        let t = self.eased_progress();
        self.from + (self.to - self.from) * t
    }
}

impl Tween<f64> {
    /// Get the current interpolated value.
    #[must_use]
    #[allow(clippy::suboptimal_flops)]
    pub fn value(&self) -> f64 {
        let t = f64::from(self.eased_progress());
        self.from + (self.to - self.from) * t
    }
}

/// Animation state for tracking animation lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationState {
    /// Animation is idle/not started
    Idle,
    /// Animation is running
    Running,
    /// Animation is paused
    Paused,
    /// Animation has completed
    Completed,
}

/// Unique identifier for an animation.
pub type AnimationId = u64;

/// An animation instance that can be managed by an Animator.
#[derive(Debug)]
pub struct AnimationInstance {
    /// Unique ID
    pub id: AnimationId,
    /// Tween for the animation
    pub tween: Tween<f32>,
    /// Current state
    pub state: AnimationState,
    /// Loop count (0 = infinite, 1 = once, N = N times)
    pub loop_count: u32,
    /// Current loop iteration
    pub current_loop: u32,
    /// Whether to reverse on alternate loops (ping-pong)
    pub alternate: bool,
    /// Direction (true = forward, false = reverse)
    forward: bool,
}

impl AnimationInstance {
    /// Create a new animation instance.
    pub fn new(id: AnimationId, from: f32, to: f32, duration_ms: u32) -> Self {
        Self {
            id,
            tween: Tween::new(from, to, duration_ms),
            state: AnimationState::Idle,
            loop_count: 1,
            current_loop: 0,
            alternate: false,
            forward: true,
        }
    }

    /// Set easing function.
    #[must_use]
    pub const fn with_easing(mut self, easing: EasingFunction) -> Self {
        self.tween = self.tween.with_easing(easing);
        self
    }

    /// Set loop count (0 = infinite).
    #[must_use]
    pub const fn with_loop_count(mut self, count: u32) -> Self {
        self.loop_count = count;
        self
    }

    /// Enable ping-pong alternating.
    #[must_use]
    pub const fn with_alternate(mut self, alternate: bool) -> Self {
        self.alternate = alternate;
        self
    }

    /// Start the animation.
    pub fn start(&mut self) {
        self.state = AnimationState::Running;
        self.current_loop = 0;
        self.forward = true;
        self.tween.reset();
    }

    /// Pause the animation.
    pub fn pause(&mut self) {
        if self.state == AnimationState::Running {
            self.state = AnimationState::Paused;
        }
    }

    /// Resume the animation.
    pub fn resume(&mut self) {
        if self.state == AnimationState::Paused {
            self.state = AnimationState::Running;
        }
    }

    /// Stop the animation.
    pub fn stop(&mut self) {
        self.state = AnimationState::Idle;
        self.tween.reset();
    }

    /// Get the current value.
    #[must_use]
    #[allow(clippy::suboptimal_flops)]
    pub fn value(&self) -> f32 {
        if self.forward {
            self.tween.value()
        } else {
            self.tween.from
                + (self.tween.to - self.tween.from) * (1.0 - self.tween.eased_progress())
        }
    }

    /// Advance the animation by delta milliseconds.
    pub fn advance(&mut self, delta_ms: u32) {
        if self.state != AnimationState::Running {
            return;
        }

        self.tween.advance(delta_ms);

        if self.tween.is_complete() {
            // Handle looping
            if self.loop_count == 0 || self.current_loop + 1 < self.loop_count {
                self.current_loop += 1;
                self.tween.reset();

                if self.alternate {
                    self.forward = !self.forward;
                }
            } else {
                self.state = AnimationState::Completed;
            }
        }
    }
}

/// Manages multiple animations.
#[derive(Debug, Default)]
pub struct Animator {
    animations: Vec<AnimationInstance>,
    next_id: AnimationId,
}

impl Animator {
    /// Create a new animator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new animation and return its ID.
    pub fn create(&mut self, from: f32, to: f32, duration_ms: u32) -> AnimationId {
        let id = self.next_id;
        self.next_id += 1;
        self.animations
            .push(AnimationInstance::new(id, from, to, duration_ms));
        id
    }

    /// Get an animation by ID.
    #[must_use]
    pub fn get(&self, id: AnimationId) -> Option<&AnimationInstance> {
        self.animations.iter().find(|a| a.id == id)
    }

    /// Get a mutable animation by ID.
    pub fn get_mut(&mut self, id: AnimationId) -> Option<&mut AnimationInstance> {
        self.animations.iter_mut().find(|a| a.id == id)
    }

    /// Start an animation.
    pub fn start(&mut self, id: AnimationId) {
        if let Some(anim) = self.get_mut(id) {
            anim.start();
        }
    }

    /// Pause an animation.
    pub fn pause(&mut self, id: AnimationId) {
        if let Some(anim) = self.get_mut(id) {
            anim.pause();
        }
    }

    /// Resume an animation.
    pub fn resume(&mut self, id: AnimationId) {
        if let Some(anim) = self.get_mut(id) {
            anim.resume();
        }
    }

    /// Stop an animation.
    pub fn stop(&mut self, id: AnimationId) {
        if let Some(anim) = self.get_mut(id) {
            anim.stop();
        }
    }

    /// Remove an animation.
    pub fn remove(&mut self, id: AnimationId) {
        self.animations.retain(|a| a.id != id);
    }

    /// Advance all animations by delta milliseconds.
    pub fn advance(&mut self, delta_ms: u32) {
        for anim in &mut self.animations {
            anim.advance(delta_ms);
        }
    }

    /// Get the value of an animation.
    #[must_use]
    pub fn value(&self, id: AnimationId) -> Option<f32> {
        self.get(id).map(AnimationInstance::value)
    }

    /// Get the number of animations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.animations.len()
    }

    /// Check if there are no animations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.animations.is_empty()
    }

    /// Remove all completed animations.
    pub fn cleanup_completed(&mut self) {
        self.animations
            .retain(|a| a.state != AnimationState::Completed);
    }

    /// Check if any animations are running.
    #[must_use]
    pub fn has_running(&self) -> bool {
        self.animations
            .iter()
            .any(|a| a.state == AnimationState::Running)
    }
}

/// A timer that fires at regular intervals.
#[derive(Debug)]
pub struct Timer {
    /// Interval in milliseconds
    pub interval_ms: u32,
    /// Elapsed time since last tick
    elapsed_ms: u32,
    /// Whether the timer is running
    running: bool,
    /// Number of times the timer has fired
    tick_count: u64,
    /// Optional limit on tick count (0 = unlimited)
    max_ticks: u64,
}

impl Timer {
    /// Create a new timer with the given interval.
    #[must_use]
    pub const fn new(interval_ms: u32) -> Self {
        Self {
            interval_ms,
            elapsed_ms: 0,
            running: false,
            tick_count: 0,
            max_ticks: 0,
        }
    }

    /// Set maximum tick count (0 = unlimited).
    #[must_use]
    pub const fn with_max_ticks(mut self, max: u64) -> Self {
        self.max_ticks = max;
        self
    }

    /// Start the timer.
    pub fn start(&mut self) {
        self.running = true;
    }

    /// Stop the timer.
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Reset the timer.
    pub fn reset(&mut self) {
        self.elapsed_ms = 0;
        self.tick_count = 0;
    }

    /// Check if the timer is running.
    #[must_use]
    pub const fn is_running(&self) -> bool {
        self.running
    }

    /// Get the tick count.
    #[must_use]
    pub const fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Advance the timer and return the number of ticks that occurred.
    pub fn advance(&mut self, delta_ms: u32) -> u32 {
        if !self.running || self.interval_ms == 0 {
            return 0;
        }

        self.elapsed_ms += delta_ms;
        let ticks = self.elapsed_ms / self.interval_ms;
        self.elapsed_ms %= self.interval_ms;

        // Apply ticks with limit check
        let mut actual_ticks = 0;
        for _ in 0..ticks {
            if self.max_ticks > 0 && self.tick_count >= self.max_ticks {
                self.running = false;
                break;
            }
            self.tick_count += 1;
            actual_ticks += 1;
        }

        actual_ticks
    }

    /// Get progress to next tick (0.0 to 1.0).
    #[must_use]
    pub fn progress(&self) -> f32 {
        if self.interval_ms == 0 {
            0.0
        } else {
            self.elapsed_ms as f32 / self.interval_ms as f32
        }
    }
}

/// Frame timer for 60fps animations.
#[derive(Debug)]
pub struct FrameTimer {
    /// Target frame duration in microseconds (16667 for 60fps)
    target_frame_us: u64,
    /// Last frame timestamp in microseconds
    last_frame_us: Option<u64>,
    /// Accumulated frame time for averaging
    frame_times: [u64; 60],
    /// Current frame index
    frame_index: usize,
    /// Number of recorded frame deltas
    delta_count: usize,
    /// Total frames rendered
    total_frames: u64,
}

impl Default for FrameTimer {
    fn default() -> Self {
        Self::new(60)
    }
}

impl FrameTimer {
    /// Create a new frame timer with target FPS.
    #[must_use]
    pub fn new(target_fps: u32) -> Self {
        let target_frame_us = if target_fps > 0 {
            1_000_000 / u64::from(target_fps)
        } else {
            16667
        };
        Self {
            target_frame_us,
            last_frame_us: None,
            frame_times: [0; 60],
            frame_index: 0,
            delta_count: 0,
            total_frames: 0,
        }
    }

    /// Record a frame with the current timestamp in microseconds.
    pub fn frame(&mut self, now_us: u64) {
        if let Some(last) = self.last_frame_us {
            let delta = now_us.saturating_sub(last);
            self.frame_times[self.frame_index] = delta;
            self.frame_index = (self.frame_index + 1) % 60;
            self.delta_count = (self.delta_count + 1).min(60);
        }
        self.last_frame_us = Some(now_us);
        self.total_frames += 1;
    }

    /// Get the average frame time in microseconds.
    #[must_use]
    pub fn average_frame_time_us(&self) -> u64 {
        if self.delta_count == 0 {
            return self.target_frame_us;
        }
        let sum: u64 = self.frame_times[..self.delta_count].iter().sum();
        sum / self.delta_count as u64
    }

    /// Get the current FPS.
    #[must_use]
    pub fn fps(&self) -> f32 {
        let avg = self.average_frame_time_us();
        if avg == 0 {
            0.0
        } else {
            1_000_000.0 / avg as f32
        }
    }

    /// Check if we're hitting target FPS (within 10% tolerance).
    #[must_use]
    pub fn is_on_target(&self) -> bool {
        let avg = self.average_frame_time_us();
        let target = self.target_frame_us;
        // Within 10% of target
        avg <= target + target / 10
    }

    /// Get the target frame time in milliseconds.
    #[must_use]
    pub fn target_frame_ms(&self) -> f32 {
        self.target_frame_us as f32 / 1000.0
    }

    /// Get total frames rendered.
    #[must_use]
    pub const fn total_frames(&self) -> u64 {
        self.total_frames
    }
}

// =============================================================================
// Data Refresh Manager
// =============================================================================

/// Manages periodic data refresh for data sources.
#[derive(Debug)]
pub struct DataRefreshManager {
    /// Registered refresh tasks
    tasks: Vec<RefreshTask>,
    /// Current timestamp in milliseconds
    current_time_ms: u64,
}

/// A scheduled data refresh task.
#[derive(Debug, Clone)]
pub struct RefreshTask {
    /// Data source key
    pub key: String,
    /// Refresh interval in milliseconds
    pub interval_ms: u64,
    /// Last refresh timestamp
    pub last_refresh_ms: u64,
    /// Whether task is active
    pub active: bool,
}

impl DataRefreshManager {
    /// Create a new refresh manager.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            tasks: Vec::new(),
            current_time_ms: 0,
        }
    }

    /// Register a data source for periodic refresh.
    ///
    /// # Arguments
    ///
    /// * `key` - Data source identifier
    /// * `interval_ms` - Refresh interval in milliseconds
    pub fn register(&mut self, key: impl Into<String>, interval_ms: u64) {
        let key = key.into();

        // Check if already registered
        if let Some(task) = self.tasks.iter_mut().find(|t| t.key == key) {
            task.interval_ms = interval_ms;
            task.active = true;
            return;
        }

        self.tasks.push(RefreshTask {
            key,
            interval_ms,
            last_refresh_ms: 0,
            active: true,
        });
    }

    /// Unregister a data source.
    pub fn unregister(&mut self, key: &str) {
        self.tasks.retain(|t| t.key != key);
    }

    /// Pause refresh for a data source.
    pub fn pause(&mut self, key: &str) {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.key == key) {
            task.active = false;
        }
    }

    /// Resume refresh for a data source.
    pub fn resume(&mut self, key: &str) {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.key == key) {
            task.active = true;
        }
    }

    /// Update the manager with the current timestamp.
    ///
    /// Returns keys of data sources that need to be refreshed.
    pub fn update(&mut self, current_time_ms: u64) -> Vec<String> {
        self.current_time_ms = current_time_ms;

        let mut to_refresh = Vec::new();

        for task in &mut self.tasks {
            if !task.active {
                continue;
            }

            let elapsed = current_time_ms.saturating_sub(task.last_refresh_ms);
            if elapsed >= task.interval_ms {
                to_refresh.push(task.key.clone());
                task.last_refresh_ms = current_time_ms;
            }
        }

        to_refresh
    }

    /// Force immediate refresh of a data source.
    pub fn force_refresh(&mut self, key: &str) -> bool {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.key == key) {
            task.last_refresh_ms = 0;
            true
        } else {
            false
        }
    }

    /// Get all registered tasks.
    #[must_use]
    pub fn tasks(&self) -> &[RefreshTask] {
        &self.tasks
    }

    /// Get task by key.
    #[must_use]
    pub fn get_task(&self, key: &str) -> Option<&RefreshTask> {
        self.tasks.iter().find(|t| t.key == key)
    }

    /// Check if a data source is due for refresh.
    #[must_use]
    pub fn is_due(&self, key: &str) -> bool {
        if let Some(task) = self.tasks.iter().find(|t| t.key == key) {
            if !task.active {
                return false;
            }
            let elapsed = self.current_time_ms.saturating_sub(task.last_refresh_ms);
            elapsed >= task.interval_ms
        } else {
            false
        }
    }

    /// Get time until next refresh for a data source (in ms).
    #[must_use]
    pub fn time_until_refresh(&self, key: &str) -> Option<u64> {
        self.tasks.iter().find(|t| t.key == key).map(|task| {
            if !task.active {
                return u64::MAX;
            }
            let elapsed = self.current_time_ms.saturating_sub(task.last_refresh_ms);
            task.interval_ms.saturating_sub(elapsed)
        })
    }
}

impl Default for DataRefreshManager {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Widget Animation API
// =============================================================================

/// Configuration for property transitions.
#[derive(Debug, Clone)]
pub struct TransitionConfig {
    /// Duration in milliseconds
    pub duration_ms: u32,
    /// Easing function
    pub easing: EasingFunction,
    /// Delay before starting in milliseconds
    pub delay_ms: u32,
}

impl Default for TransitionConfig {
    fn default() -> Self {
        Self {
            duration_ms: 300,
            easing: EasingFunction::EaseInOutCubic,
            delay_ms: 0,
        }
    }
}

impl TransitionConfig {
    /// Create a new transition configuration.
    #[must_use]
    pub const fn new(duration_ms: u32) -> Self {
        Self {
            duration_ms,
            easing: EasingFunction::EaseInOutCubic,
            delay_ms: 0,
        }
    }

    /// Set the easing function.
    #[must_use]
    pub const fn with_easing(mut self, easing: EasingFunction) -> Self {
        self.easing = easing;
        self
    }

    /// Set the delay.
    #[must_use]
    pub const fn with_delay(mut self, delay_ms: u32) -> Self {
        self.delay_ms = delay_ms;
        self
    }

    /// Quick preset (150ms)
    #[must_use]
    pub const fn quick() -> Self {
        Self::new(150)
    }

    /// Normal preset (300ms)
    #[must_use]
    pub const fn normal() -> Self {
        Self::new(300)
    }

    /// Slow preset (500ms)
    #[must_use]
    pub const fn slow() -> Self {
        Self::new(500)
    }
}

/// An animated property that smoothly transitions between values.
///
/// Use this in widget state to animate property changes automatically.
#[derive(Debug, Clone)]
pub struct AnimatedProperty<T> {
    /// Current visual value (what's rendered)
    current: T,
    /// Target value we're animating towards
    target: T,
    /// Starting value of current animation
    start: T,
    /// Transition configuration
    config: TransitionConfig,
    /// Elapsed time in milliseconds
    elapsed_ms: u32,
    /// Whether an animation is in progress
    animating: bool,
}

impl<T: Clone + Default> Default for AnimatedProperty<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: Clone> AnimatedProperty<T> {
    /// Create a new animated property with an initial value.
    pub fn new(value: T) -> Self {
        Self {
            current: value.clone(),
            target: value.clone(),
            start: value,
            config: TransitionConfig::default(),
            elapsed_ms: 0,
            animating: false,
        }
    }

    /// Create with a custom transition config.
    pub fn with_config(value: T, config: TransitionConfig) -> Self {
        Self {
            current: value.clone(),
            target: value.clone(),
            start: value,
            config,
            elapsed_ms: 0,
            animating: false,
        }
    }

    /// Get the current visual value.
    pub const fn get(&self) -> &T {
        &self.current
    }

    /// Get the target value.
    pub const fn target(&self) -> &T {
        &self.target
    }

    /// Check if currently animating.
    #[must_use]
    pub const fn is_animating(&self) -> bool {
        self.animating
    }

    /// Set a new target value, starting an animation.
    pub fn set(&mut self, value: T) {
        self.start = self.current.clone();
        self.target = value;
        self.elapsed_ms = 0;
        self.animating = true;
    }

    /// Set value immediately without animation.
    pub fn set_immediate(&mut self, value: T) {
        self.current = value.clone();
        self.target = value.clone();
        self.start = value;
        self.animating = false;
        self.elapsed_ms = 0;
    }

    /// Get animation progress (0.0 to 1.0).
    #[must_use]
    pub fn progress(&self) -> f32 {
        if !self.animating {
            return 1.0;
        }

        let total = self.config.duration_ms + self.config.delay_ms;
        if total == 0 {
            return 1.0;
        }

        if self.elapsed_ms < self.config.delay_ms {
            return 0.0;
        }

        let elapsed_after_delay = self.elapsed_ms - self.config.delay_ms;
        (elapsed_after_delay as f32 / self.config.duration_ms as f32).min(1.0)
    }

    /// Get eased progress.
    #[must_use]
    pub fn eased_progress(&self) -> f32 {
        self.config.easing.apply(self.progress())
    }
}

impl AnimatedProperty<f32> {
    /// Advance the animation by delta milliseconds.
    pub fn advance(&mut self, delta_ms: u32) {
        if !self.animating {
            return;
        }

        self.elapsed_ms += delta_ms;

        let t = self.eased_progress();
        self.current = (self.target - self.start).mul_add(t, self.start);

        if self.progress() >= 1.0 {
            self.current = self.target;
            self.animating = false;
        }
    }
}

impl AnimatedProperty<f64> {
    /// Advance the animation by delta milliseconds.
    pub fn advance(&mut self, delta_ms: u32) {
        if !self.animating {
            return;
        }

        self.elapsed_ms += delta_ms;

        let t = f64::from(self.eased_progress());
        self.current = (self.target - self.start).mul_add(t, self.start);

        if self.progress() >= 1.0 {
            self.current = self.target;
            self.animating = false;
        }
    }
}

impl AnimatedProperty<crate::Color> {
    /// Advance the animation by delta milliseconds.
    pub fn advance(&mut self, delta_ms: u32) {
        if !self.animating {
            return;
        }

        self.elapsed_ms += delta_ms;

        let t = self.eased_progress();
        self.current = crate::Color {
            r: (self.target.r - self.start.r).mul_add(t, self.start.r),
            g: (self.target.g - self.start.g).mul_add(t, self.start.g),
            b: (self.target.b - self.start.b).mul_add(t, self.start.b),
            a: (self.target.a - self.start.a).mul_add(t, self.start.a),
        };

        if self.progress() >= 1.0 {
            self.current = self.target;
            self.animating = false;
        }
    }
}

impl AnimatedProperty<crate::Point> {
    /// Advance the animation by delta milliseconds.
    pub fn advance(&mut self, delta_ms: u32) {
        if !self.animating {
            return;
        }

        self.elapsed_ms += delta_ms;

        let t = self.eased_progress();
        self.current = crate::Point {
            x: (self.target.x - self.start.x).mul_add(t, self.start.x),
            y: (self.target.y - self.start.y).mul_add(t, self.start.y),
        };

        if self.progress() >= 1.0 {
            self.current = self.target;
            self.animating = false;
        }
    }
}

impl AnimatedProperty<crate::Size> {
    /// Advance the animation by delta milliseconds.
    pub fn advance(&mut self, delta_ms: u32) {
        if !self.animating {
            return;
        }

        self.elapsed_ms += delta_ms;

        let t = self.eased_progress();
        self.current = crate::Size {
            width: (self.target.width - self.start.width).mul_add(t, self.start.width),
            height: (self.target.height - self.start.height).mul_add(t, self.start.height),
        };

        if self.progress() >= 1.0 {
            self.current = self.target;
            self.animating = false;
        }
    }
}

/// Spring animation configuration.
#[derive(Debug, Clone, Copy)]
pub struct SpringConfig {
    /// Spring stiffness (higher = faster oscillation)
    pub stiffness: f32,
    /// Damping (higher = less oscillation)
    pub damping: f32,
    /// Mass of the object
    pub mass: f32,
}

impl Default for SpringConfig {
    fn default() -> Self {
        Self {
            stiffness: 100.0,
            damping: 10.0,
            mass: 1.0,
        }
    }
}

impl SpringConfig {
    /// Create a new spring configuration.
    #[must_use]
    pub const fn new(stiffness: f32, damping: f32, mass: f32) -> Self {
        Self {
            stiffness,
            damping,
            mass,
        }
    }

    /// Gentle spring preset.
    #[must_use]
    pub const fn gentle() -> Self {
        Self::new(100.0, 15.0, 1.0)
    }

    /// Bouncy spring preset.
    #[must_use]
    pub const fn bouncy() -> Self {
        Self::new(300.0, 10.0, 1.0)
    }

    /// Stiff spring preset.
    #[must_use]
    pub const fn stiff() -> Self {
        Self::new(500.0, 30.0, 1.0)
    }
}

/// Spring-based animation for physics-like motion.
#[derive(Debug, Clone)]
pub struct SpringAnimation {
    /// Current position
    position: f32,
    /// Current velocity
    velocity: f32,
    /// Target position
    target: f32,
    /// Spring configuration
    config: SpringConfig,
    /// Velocity threshold for considering animation complete
    velocity_threshold: f32,
    /// Position threshold for considering animation complete
    position_threshold: f32,
}

impl SpringAnimation {
    /// Create a new spring animation.
    #[must_use]
    pub fn new(initial: f32) -> Self {
        Self {
            position: initial,
            velocity: 0.0,
            target: initial,
            config: SpringConfig::default(),
            velocity_threshold: 0.01,
            position_threshold: 0.001,
        }
    }

    /// Create with custom spring config.
    #[must_use]
    pub const fn with_config(initial: f32, config: SpringConfig) -> Self {
        Self {
            position: initial,
            velocity: 0.0,
            target: initial,
            config,
            velocity_threshold: 0.01,
            position_threshold: 0.001,
        }
    }

    /// Get the current position.
    #[must_use]
    pub const fn position(&self) -> f32 {
        self.position
    }

    /// Get the current velocity.
    #[must_use]
    pub const fn velocity(&self) -> f32 {
        self.velocity
    }

    /// Get the target.
    #[must_use]
    pub const fn target(&self) -> f32 {
        self.target
    }

    /// Set the target position.
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    /// Set position immediately without animation.
    pub fn set_immediate(&mut self, position: f32) {
        self.position = position;
        self.target = position;
        self.velocity = 0.0;
    }

    /// Check if the animation is at rest.
    #[must_use]
    pub fn is_at_rest(&self) -> bool {
        let position_diff = (self.position - self.target).abs();
        let velocity_abs = self.velocity.abs();
        position_diff < self.position_threshold && velocity_abs < self.velocity_threshold
    }

    /// Advance the spring animation by delta seconds.
    pub fn advance(&mut self, delta_s: f32) {
        if self.is_at_rest() {
            self.position = self.target;
            self.velocity = 0.0;
            return;
        }

        // Spring physics: F = -kx - cv
        // a = F/m = (-kx - cv) / m
        let displacement = self.position - self.target;
        let spring_force = -self.config.stiffness * displacement;
        let damping_force = -self.config.damping * self.velocity;
        let acceleration = (spring_force + damping_force) / self.config.mass;

        // Semi-implicit Euler integration
        self.velocity += acceleration * delta_s;
        self.position += self.velocity * delta_s;
    }

    /// Advance by delta milliseconds.
    pub fn advance_ms(&mut self, delta_ms: u32) {
        self.advance(delta_ms as f32 / 1000.0);
    }
}
