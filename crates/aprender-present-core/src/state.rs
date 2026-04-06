//! State management for Presentar applications.
//!
//! This module implements the Elm Architecture pattern for predictable state
//! management: `State + Message → (State, Command)`.
//!
//! # Examples
//!
//! ```
//! use presentar_core::{State, Command};
//! use serde::{Deserialize, Serialize};
//!
//! // Define your application state
//! #[derive(Clone, Default, Serialize, Deserialize)]
//! struct AppState {
//!     count: i32,
//! }
//!
//! // Define messages that modify state
//! enum AppMessage {
//!     Increment,
//!     Decrement,
//!     Reset,
//! }
//!
//! impl State for AppState {
//!     type Message = AppMessage;
//!
//!     fn update(&mut self, msg: Self::Message) -> Command<Self::Message> {
//!         match msg {
//!             AppMessage::Increment => self.count += 1,
//!             AppMessage::Decrement => self.count -= 1,
//!             AppMessage::Reset => self.count = 0,
//!         }
//!         Command::None
//!     }
//! }
//!
//! let mut state = AppState::default();
//! state.update(AppMessage::Increment);
//! assert_eq!(state.count, 1);
//! ```

use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;

/// Application state trait.
///
/// Implements the Elm Architecture: State + Message → (State, Command)
pub trait State: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync {
    /// Message type for state updates
    type Message: Send;

    /// Update state in response to a message.
    ///
    /// Returns a command for side effects (async operations, navigation, etc.)
    fn update(&mut self, msg: Self::Message) -> Command<Self::Message>;
}

/// Commands for side effects.
///
/// Commands represent effects that should happen after a state update:
/// - Async tasks (data fetching, file operations)
/// - Navigation
/// - State persistence
#[derive(Default)]
pub enum Command<M> {
    /// No command
    #[default]
    None,
    /// Execute multiple commands
    Batch(Vec<Self>),
    /// Execute an async task
    Task(Pin<Box<dyn Future<Output = M> + Send>>),
    /// Navigate to a route
    Navigate {
        /// Route path
        route: String,
    },
    /// Save state to storage
    SaveState {
        /// Storage key
        key: String,
    },
    /// Load state from storage
    LoadState {
        /// Storage key
        key: String,
        /// Callback with loaded state
        on_load: fn(Option<Vec<u8>>) -> M,
    },
}

impl<M> Command<M> {
    /// Create a task command from an async block.
    pub fn task<F>(future: F) -> Self
    where
        F: Future<Output = M> + Send + 'static,
    {
        Self::Task(Box::pin(future))
    }

    /// Create a batch of commands.
    pub fn batch(commands: impl IntoIterator<Item = Self>) -> Self {
        Self::Batch(commands.into_iter().collect())
    }

    /// Check if this is the none command.
    #[must_use]
    pub const fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Map the message type using a function.
    pub fn map<N, F>(self, f: F) -> Command<N>
    where
        F: Fn(M) -> N + Send + Sync + 'static,
        M: Send + 'static,
        N: Send + 'static,
    {
        let f: std::sync::Arc<dyn Fn(M) -> N + Send + Sync> = std::sync::Arc::new(f);
        self.map_inner(&f)
    }

    fn map_inner<N>(self, f: &std::sync::Arc<dyn Fn(M) -> N + Send + Sync>) -> Command<N>
    where
        M: Send + 'static,
        N: Send + 'static,
    {
        match self {
            Self::None => Command::None,
            Self::Batch(cmds) => Command::Batch(cmds.into_iter().map(|c| c.map_inner(f)).collect()),
            Self::Task(fut) => {
                let f = f.clone();
                Command::Task(Box::pin(async move { f(fut.await) }))
            }
            Self::Navigate { route } => Command::Navigate { route },
            Self::SaveState { key } => Command::SaveState { key },
            Self::LoadState { .. } => {
                // Can't easily map LoadState due to function pointer
                // In practice, LoadState is usually at the top level
                Command::None
            }
        }
    }
}

/// A simple counter state for testing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CounterState {
    /// Current count
    pub count: i32,
}

/// Messages for the counter state.
#[derive(Debug, Clone)]
pub enum CounterMessage {
    /// Increment the counter
    Increment,
    /// Decrement the counter
    Decrement,
    /// Set the counter to a specific value
    Set(i32),
    /// Reset to zero
    Reset,
}

impl State for CounterState {
    type Message = CounterMessage;

    fn update(&mut self, msg: Self::Message) -> Command<Self::Message> {
        match msg {
            CounterMessage::Increment => self.count += 1,
            CounterMessage::Decrement => self.count -= 1,
            CounterMessage::Set(value) => self.count = value,
            CounterMessage::Reset => self.count = 0,
        }
        Command::None
    }
}

/// Type alias for state change subscribers.
type Subscriber<S> = Box<dyn Fn(&S) + Send + Sync>;

/// Store manages state lifecycle with subscriptions and time-travel debugging.
pub struct Store<S: State> {
    state: S,
    history: Vec<S>,
    history_index: usize,
    max_history: usize,
    subscribers: Vec<Subscriber<S>>,
}

impl<S: State> Store<S> {
    /// Create a new store with initial state.
    pub fn new(initial: S) -> Self {
        Self {
            state: initial,
            history: Vec::new(),
            history_index: 0,
            max_history: 100,
            subscribers: Vec::new(),
        }
    }

    /// Create a store with custom history limit.
    pub fn with_history_limit(initial: S, max_history: usize) -> Self {
        Self {
            state: initial,
            history: Vec::new(),
            history_index: 0,
            max_history,
            subscribers: Vec::new(),
        }
    }

    /// Get current state.
    pub const fn state(&self) -> &S {
        &self.state
    }

    /// Dispatch a message to update state.
    pub fn dispatch(&mut self, msg: S::Message) -> Command<S::Message> {
        // Save current state to history
        if self.max_history > 0 {
            // Truncate future history if we're not at the end
            if self.history_index < self.history.len() {
                self.history.truncate(self.history_index);
            }

            self.history.push(self.state.clone());

            // Limit history size
            if self.history.len() > self.max_history {
                self.history.remove(0);
            } else {
                self.history_index = self.history.len();
            }
        }

        // Update state
        let cmd = self.state.update(msg);

        // Notify subscribers
        self.notify_subscribers();

        cmd
    }

    /// Subscribe to state changes.
    pub fn subscribe<F>(&mut self, callback: F)
    where
        F: Fn(&S) + Send + Sync + 'static,
    {
        self.subscribers.push(Box::new(callback));
    }

    /// Get number of history entries.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Can undo to previous state.
    pub const fn can_undo(&self) -> bool {
        self.history_index > 0
    }

    /// Can redo to next state.
    pub fn can_redo(&self) -> bool {
        self.history_index < self.history.len()
    }

    /// Undo to previous state.
    pub fn undo(&mut self) -> bool {
        if self.can_undo() {
            // If we're at the end, save current state first
            if self.history_index == self.history.len() {
                self.history.push(self.state.clone());
            }

            self.history_index -= 1;
            self.state = self.history[self.history_index].clone();
            self.notify_subscribers();
            true
        } else {
            false
        }
    }

    /// Redo to next state.
    pub fn redo(&mut self) -> bool {
        if self.history_index < self.history.len().saturating_sub(1) {
            self.history_index += 1;
            self.state = self.history[self.history_index].clone();
            self.notify_subscribers();
            true
        } else {
            false
        }
    }

    /// Jump to a specific point in history.
    pub fn jump_to(&mut self, index: usize) -> bool {
        if index < self.history.len() {
            self.history_index = index;
            self.state = self.history[index].clone();
            self.notify_subscribers();
            true
        } else {
            false
        }
    }

    /// Clear history.
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.history_index = 0;
    }

    fn notify_subscribers(&self) {
        for subscriber in &self.subscribers {
            subscriber(&self.state);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_increment() {
        let mut state = CounterState::default();
        state.update(CounterMessage::Increment);
        assert_eq!(state.count, 1);
    }

    #[test]
    fn test_counter_decrement() {
        let mut state = CounterState { count: 5 };
        state.update(CounterMessage::Decrement);
        assert_eq!(state.count, 4);
    }

    #[test]
    fn test_counter_set() {
        let mut state = CounterState::default();
        state.update(CounterMessage::Set(42));
        assert_eq!(state.count, 42);
    }

    #[test]
    fn test_counter_reset() {
        let mut state = CounterState { count: 100 };
        state.update(CounterMessage::Reset);
        assert_eq!(state.count, 0);
    }

    #[test]
    fn test_command_none() {
        let cmd: Command<()> = Command::None;
        assert!(cmd.is_none());
    }

    #[test]
    fn test_command_default() {
        let cmd: Command<()> = Command::default();
        assert!(cmd.is_none());
    }

    #[test]
    fn test_command_batch() {
        let cmd: Command<i32> = Command::batch([
            Command::Navigate {
                route: "/a".to_string(),
            },
            Command::Navigate {
                route: "/b".to_string(),
            },
        ]);
        assert!(!cmd.is_none());
        if let Command::Batch(cmds) = cmd {
            assert_eq!(cmds.len(), 2);
        } else {
            panic!("Expected Batch command");
        }
    }

    #[test]
    fn test_command_navigate() {
        let cmd: Command<()> = Command::Navigate {
            route: "/home".to_string(),
        };
        if let Command::Navigate { route } = cmd {
            assert_eq!(route, "/home");
        } else {
            panic!("Expected Navigate command");
        }
    }

    #[test]
    fn test_command_save_state() {
        let cmd: Command<()> = Command::SaveState {
            key: "app_state".to_string(),
        };
        if let Command::SaveState { key } = cmd {
            assert_eq!(key, "app_state");
        } else {
            panic!("Expected SaveState command");
        }
    }

    #[test]
    fn test_counter_serialization() {
        let state = CounterState { count: 42 };
        let json = serde_json::to_string(&state).unwrap();
        let loaded: CounterState = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.count, 42);
    }

    #[test]
    fn test_command_map() {
        let cmd: Command<i32> = Command::Navigate {
            route: "/test".to_string(),
        };
        let mapped: Command<String> = cmd.map(|_i| "mapped".to_string());

        if let Command::Navigate { route } = mapped {
            assert_eq!(route, "/test");
        } else {
            panic!("Expected Navigate command after map");
        }
    }

    #[test]
    fn test_command_map_none() {
        let cmd: Command<i32> = Command::None;
        let mapped: Command<String> = cmd.map(|i| i.to_string());
        assert!(mapped.is_none());
    }

    #[test]
    fn test_command_batch_map() {
        let cmd: Command<i32> = Command::batch([
            Command::SaveState {
                key: "key1".to_string(),
            },
            Command::SaveState {
                key: "key2".to_string(),
            },
        ]);

        let mapped: Command<String> = cmd.map(|i| format!("val_{i}"));

        if let Command::Batch(cmds) = mapped {
            assert_eq!(cmds.len(), 2);
        } else {
            panic!("Expected Batch command after map");
        }
    }

    // =========================================================================
    // Store Tests
    // =========================================================================

    #[test]
    fn test_store_new() {
        let store = Store::new(CounterState::default());
        assert_eq!(store.state().count, 0);
    }

    #[test]
    fn test_store_dispatch() {
        let mut store = Store::new(CounterState::default());
        store.dispatch(CounterMessage::Increment);
        assert_eq!(store.state().count, 1);
    }

    #[test]
    fn test_store_history() {
        let mut store = Store::new(CounterState::default());

        store.dispatch(CounterMessage::Increment);
        store.dispatch(CounterMessage::Increment);
        store.dispatch(CounterMessage::Increment);

        assert_eq!(store.state().count, 3);
        assert_eq!(store.history_len(), 3);
    }

    #[test]
    fn test_store_undo() {
        let mut store = Store::new(CounterState::default());

        store.dispatch(CounterMessage::Increment);
        store.dispatch(CounterMessage::Increment);
        assert_eq!(store.state().count, 2);

        assert!(store.can_undo());
        assert!(store.undo());
        assert_eq!(store.state().count, 1);

        assert!(store.undo());
        assert_eq!(store.state().count, 0);
    }

    #[test]
    fn test_store_redo() {
        let mut store = Store::new(CounterState::default());

        store.dispatch(CounterMessage::Increment);
        store.dispatch(CounterMessage::Increment);
        store.undo();
        store.undo();

        assert_eq!(store.state().count, 0);
        assert!(store.can_redo());

        assert!(store.redo());
        assert_eq!(store.state().count, 1);

        assert!(store.redo());
        assert_eq!(store.state().count, 2);
    }

    #[test]
    fn test_store_undo_at_start() {
        let mut store = Store::new(CounterState::default());
        assert!(!store.can_undo());
        assert!(!store.undo());
    }

    #[test]
    fn test_store_redo_at_end() {
        let mut store = Store::new(CounterState::default());
        store.dispatch(CounterMessage::Increment);
        assert!(!store.can_redo());
        assert!(!store.redo());
    }

    #[test]
    fn test_store_history_truncation() {
        let mut store = Store::new(CounterState::default());

        store.dispatch(CounterMessage::Set(1));
        store.dispatch(CounterMessage::Set(2));
        store.dispatch(CounterMessage::Set(3));

        // Undo to 1
        store.undo();
        store.undo();
        assert_eq!(store.state().count, 1);

        // New dispatch should truncate redo history
        store.dispatch(CounterMessage::Set(10));
        assert_eq!(store.state().count, 10);

        // Cannot redo to 2 or 3 anymore
        assert!(!store.redo());
    }

    #[test]
    fn test_store_jump_to() {
        let mut store = Store::new(CounterState::default());

        store.dispatch(CounterMessage::Set(10));
        store.dispatch(CounterMessage::Set(20));
        store.dispatch(CounterMessage::Set(30));

        assert!(store.jump_to(0));
        assert_eq!(store.state().count, 0);

        assert!(store.jump_to(2));
        assert_eq!(store.state().count, 20);
    }

    #[test]
    fn test_store_jump_invalid() {
        let mut store = Store::new(CounterState::default());
        store.dispatch(CounterMessage::Increment);

        assert!(!store.jump_to(100));
    }

    #[test]
    fn test_store_clear_history() {
        let mut store = Store::new(CounterState::default());

        store.dispatch(CounterMessage::Increment);
        store.dispatch(CounterMessage::Increment);
        assert!(store.history_len() > 0);

        store.clear_history();
        assert_eq!(store.history_len(), 0);
        assert!(!store.can_undo());
    }

    #[test]
    fn test_store_with_history_limit() {
        let mut store = Store::with_history_limit(CounterState::default(), 3);

        for i in 1..=10 {
            store.dispatch(CounterMessage::Set(i));
        }

        // History should be capped at 3
        assert!(store.history_len() <= 3);
    }

    #[test]
    fn test_store_subscribe() {
        use std::sync::atomic::{AtomicI32, Ordering};
        use std::sync::Arc;

        let call_count = Arc::new(AtomicI32::new(0));
        let call_count_clone = call_count.clone();

        let mut store = Store::new(CounterState::default());
        store.subscribe(move |_| {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
        });

        store.dispatch(CounterMessage::Increment);
        store.dispatch(CounterMessage::Increment);

        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_store_no_history() {
        let mut store = Store::with_history_limit(CounterState::default(), 0);

        store.dispatch(CounterMessage::Increment);
        store.dispatch(CounterMessage::Increment);

        assert_eq!(store.history_len(), 0);
        assert!(!store.can_undo());
    }
}
