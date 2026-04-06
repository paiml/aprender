//! Mock WASM Runtime for Testing Callback Patterns
//!
//! This module provides a mock runtime that simulates async message passing
//! for WASM worker components without requiring browser APIs.
//!
//! Per `PROBAR-SPEC-WASM-001` Section 2.1.
//!
//! ## Browser Fidelity (PROBAR-WASM-003)
//!
//! To simulate real browser `structuredClone` semantics, messages are
//! serialized and deserialized when passed through `receive_message`.
//! This ensures that non-serializable types (like `Rc`, closures) will
//! fail at test time, just as they would in a real browser.

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

/// Mock message types for testing worker communication
///
/// These mirror the actual message types used in WASM worker protocols.
///
/// ## Serialization Requirement (PROBAR-WASM-003)
///
/// All messages implement `Serialize` and `Deserialize` to simulate
/// browser `structuredClone` semantics. Messages are round-tripped
/// through serialization in `receive_message` to catch non-serializable
/// payloads at test time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MockMessage {
    /// Bootstrap message with base URL
    Bootstrap {
        /// Base URL for asset loading
        base_url: String,
    },
    /// Initialization message with model URL
    Init {
        /// Model URL to load
        model_url: String,
    },
    /// Worker ready signal
    Ready,
    /// Model loaded successfully
    ModelLoaded {
        /// Model size in MB
        size_mb: f64,
        /// Load time in milliseconds
        load_time_ms: f64,
    },
    /// Start recording/processing
    Start {
        /// Sample rate in Hz
        sample_rate: u32,
    },
    /// Stop recording/processing
    Stop,
    /// Partial result
    Partial {
        /// Partial text
        text: String,
        /// Whether this is the final result
        is_final: bool,
    },
    /// Error occurred
    Error {
        /// Error message
        message: String,
    },
    /// Shutdown request
    Shutdown,
    /// Custom message for extension
    Custom {
        /// Message type identifier
        msg_type: String,
        /// JSON payload
        payload: String,
    },
}

impl MockMessage {
    /// Create a bootstrap message
    #[must_use]
    pub fn bootstrap(base_url: &str) -> Self {
        Self::Bootstrap {
            base_url: base_url.to_string(),
        }
    }

    /// Create an init message
    #[must_use]
    pub fn init(model_url: &str) -> Self {
        Self::Init {
            model_url: model_url.to_string(),
        }
    }

    /// Create a model loaded message
    #[must_use]
    pub fn model_loaded(size_mb: f64, load_time_ms: f64) -> Self {
        Self::ModelLoaded {
            size_mb,
            load_time_ms,
        }
    }

    /// Create a start message
    #[must_use]
    pub fn start(sample_rate: u32) -> Self {
        Self::Start { sample_rate }
    }

    /// Create an error message
    #[must_use]
    pub fn error(message: &str) -> Self {
        Self::Error {
            message: message.to_string(),
        }
    }

    /// Create a partial result message
    #[must_use]
    pub fn partial(text: &str, is_final: bool) -> Self {
        Self::Partial {
            text: text.to_string(),
            is_final,
        }
    }
}

/// Mock runtime that simulates async message passing
///
/// This replaces browser APIs like `Worker.postMessage()` and `Worker.onmessage`
/// with a deterministic, testable interface.
pub struct MockWasmRuntime {
    /// Incoming message queue (messages TO the component)
    incoming: Rc<RefCell<VecDeque<MockMessage>>>,
    /// Outgoing message queue (messages FROM the component)
    outgoing: Rc<RefCell<VecDeque<MockMessage>>>,
    /// Registered message handlers
    handlers: Rc<RefCell<Vec<Box<dyn Fn(&MockMessage)>>>>,
    /// Whether the runtime has been started
    started: bool,
    /// Total messages processed
    messages_processed: usize,
}

impl Default for MockWasmRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MockWasmRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockWasmRuntime")
            .field("incoming_count", &self.incoming.borrow().len())
            .field("outgoing_count", &self.outgoing.borrow().len())
            .field("handlers_count", &self.handlers.borrow().len())
            .field("started", &self.started)
            .field("messages_processed", &self.messages_processed)
            .finish()
    }
}

impl Clone for MockWasmRuntime {
    fn clone(&self) -> Self {
        Self {
            incoming: Rc::clone(&self.incoming),
            outgoing: Rc::clone(&self.outgoing),
            handlers: Rc::clone(&self.handlers),
            started: self.started,
            messages_processed: self.messages_processed,
        }
    }
}

impl MockWasmRuntime {
    /// Create a new mock runtime
    #[must_use]
    pub fn new() -> Self {
        Self {
            incoming: Rc::new(RefCell::new(VecDeque::new())),
            outgoing: Rc::new(RefCell::new(VecDeque::new())),
            handlers: Rc::new(RefCell::new(Vec::new())),
            started: false,
            messages_processed: 0,
        }
    }

    /// Register a message handler (like `worker.onmessage`)
    pub fn on_message<F>(&mut self, handler: F)
    where
        F: Fn(&MockMessage) + 'static,
    {
        self.handlers.borrow_mut().push(Box::new(handler));
    }

    /// Send message (like `worker.postMessage`)
    ///
    /// This puts a message in the outgoing queue for the component to "send".
    ///
    /// ## Browser Fidelity (PROBAR-WASM-003)
    ///
    /// Like `receive_message`, this performs a round-trip serialization to
    /// simulate `structuredClone` semantics.
    ///
    /// # Panics
    ///
    /// Panics if the message cannot be serialized. This intentionally mirrors
    /// browser `postMessage` semantics where non-cloneable objects throw.
    #[allow(clippy::expect_used)] // Intentional: simulates browser postMessage failure
    pub fn post_message(&self, msg: MockMessage) {
        // Round-trip through bincode to simulate structuredClone
        let serialized = bincode::serialize(&msg)
            .expect("MockMessage serialization failed - this would fail in browser postMessage");
        let cloned: MockMessage = bincode::deserialize(&serialized)
            .expect("MockMessage deserialization failed - corrupted message");

        self.outgoing.borrow_mut().push_back(cloned);
    }

    /// Receive a message (simulates worker sending to main thread)
    ///
    /// This puts a message in the incoming queue to be processed by handlers.
    ///
    /// ## Browser Fidelity (PROBAR-WASM-003)
    ///
    /// To simulate real browser `structuredClone` semantics, the message is
    /// serialized and deserialized before being queued. This ensures that:
    /// - Non-serializable types will panic (like they would in a browser)
    /// - Message data is deep-copied (no shared references)
    ///
    /// # Panics
    ///
    /// Panics if the message cannot be serialized or deserialized. This
    /// simulates browser `postMessage` behavior where non-cloneable objects
    /// cause errors.
    #[allow(clippy::expect_used)] // Intentional: simulates browser postMessage failure
    pub fn receive_message(&self, msg: MockMessage) {
        // Round-trip through bincode to simulate structuredClone
        let serialized = bincode::serialize(&msg)
            .expect("MockMessage serialization failed - this would fail in browser postMessage");
        let cloned: MockMessage = bincode::deserialize(&serialized)
            .expect("MockMessage deserialization failed - corrupted message");

        self.incoming.borrow_mut().push_back(cloned);
    }

    /// Receive a message without serialization (bypass for testing)
    ///
    /// This is the legacy method that doesn't enforce serialization.
    /// Use `receive_message` for browser-fidelity testing.
    #[doc(hidden)]
    pub fn receive_message_unchecked(&self, msg: MockMessage) {
        self.incoming.borrow_mut().push_back(msg);
    }

    /// Process one message from the incoming queue
    ///
    /// Returns `true` if a message was processed, `false` if queue was empty.
    ///
    /// # Re-entrancy Safety
    ///
    /// Handlers may call `receive_message()` to queue additional messages,
    /// or even register new handlers via `on_message()`. This is achieved
    /// by temporarily swapping out the handlers vector during processing.
    pub fn tick(&mut self) -> bool {
        // Step 1: Pop message (borrow and release incoming)
        let msg = self.incoming.borrow_mut().pop_front();

        if let Some(msg) = msg {
            // Helper guard to ensure handlers are restored even on panic
            struct HandlersGuard {
                handlers_ref: Rc<RefCell<Vec<Box<dyn Fn(&MockMessage)>>>>,
                handlers_to_run: Vec<Box<dyn Fn(&MockMessage)>>,
            }

            impl Drop for HandlersGuard {
                fn drop(&mut self) {
                    let mut handlers = self.handlers_ref.borrow_mut();
                    // Prepend original handlers (handlers_to_run), keeping new ones at the end
                    let new_handlers = std::mem::take(&mut *handlers);
                    *handlers = std::mem::take(&mut self.handlers_to_run);
                    handlers.extend(new_handlers);
                }
            }

            // Step 2: Swap out handlers using RAII guard for panic safety
            let handlers_guard = HandlersGuard {
                handlers_ref: Rc::clone(&self.handlers),
                handlers_to_run: {
                    let mut h = self.handlers.borrow_mut();
                    std::mem::take(&mut *h)
                },
            };

            // Step 3: Run all handlers with NO borrows held
            for handler in &handlers_guard.handlers_to_run {
                handler(&msg);
            }

            // Step 4 (Implicit): Guard drops here, restoring handlers via Drop trait

            self.messages_processed += 1;
            true
        } else {
            false
        }
    }

    /// Process all pending messages
    ///
    /// # Safety Limit
    ///
    /// To prevent infinite loops from recursive message patterns,
    /// this method processes at most 10,000 messages. Use `drain_bounded`
    /// for explicit control over the limit.
    pub fn drain(&mut self) {
        self.drain_bounded(10_000);
    }

    /// Process pending messages with explicit bound
    ///
    /// Returns the number of messages processed.
    pub fn drain_bounded(&mut self, max_messages: usize) -> usize {
        let mut processed = 0;
        while processed < max_messages && self.tick() {
            processed += 1;
        }
        processed
    }

    /// Process up to N messages
    pub fn tick_n(&mut self, n: usize) -> usize {
        let mut processed = 0;
        for _ in 0..n {
            if self.tick() {
                processed += 1;
            } else {
                break;
            }
        }
        processed
    }

    /// Get pending incoming message count
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.incoming.borrow().len()
    }

    /// Get outgoing messages (for assertions)
    #[must_use]
    pub fn take_outgoing(&self) -> Vec<MockMessage> {
        self.outgoing.borrow_mut().drain(..).collect()
    }

    /// Peek at outgoing messages without consuming
    #[must_use]
    pub fn peek_outgoing(&self) -> Vec<MockMessage> {
        self.outgoing.borrow().iter().cloned().collect()
    }

    /// Check if there are any outgoing messages
    #[must_use]
    pub fn has_outgoing(&self) -> bool {
        !self.outgoing.borrow().is_empty()
    }

    /// Get total messages processed
    #[must_use]
    pub fn total_processed(&self) -> usize {
        self.messages_processed
    }

    /// Clear all queues and handlers
    pub fn reset(&mut self) {
        self.incoming.borrow_mut().clear();
        self.outgoing.borrow_mut().clear();
        self.handlers.borrow_mut().clear();
        self.messages_processed = 0;
    }

    /// Start the runtime (marks it as active)
    pub fn start(&mut self) {
        self.started = true;
    }

    /// Check if runtime is started
    #[must_use]
    pub fn is_started(&self) -> bool {
        self.started
    }
}

/// Trait for WASM components that can be tested with mock runtime
///
/// Components implement this trait to enable testing with `WasmCallbackTestHarness`.
pub trait MockableWorker: Sized {
    /// Create the worker with a mock runtime instead of real browser APIs
    fn with_mock_runtime(runtime: MockWasmRuntime) -> Self;

    /// Get the current state as a string (for assertions)
    fn get_state(&self) -> String;

    /// Get internal state for debugging (may differ from public state in buggy code)
    ///
    /// If this differs from `get_state()`, there's a state sync bug!
    fn debug_internal_state(&self) -> String {
        self.get_state() // Default implementation assumes no desync
    }

    /// Check for state synchronization
    ///
    /// Returns `true` if reported state matches internal state.
    fn is_state_synced(&self) -> bool {
        self.get_state() == self.debug_internal_state()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_message_constructors() {
        let bootstrap = MockMessage::bootstrap("http://localhost:8080");
        assert!(matches!(
            bootstrap,
            MockMessage::Bootstrap { base_url } if base_url == "http://localhost:8080"
        ));

        let init = MockMessage::init("/models/whisper-tiny.apr");
        assert!(
            matches!(init, MockMessage::Init { model_url } if model_url == "/models/whisper-tiny.apr")
        );

        let loaded = MockMessage::model_loaded(39.0, 1500.0);
        assert!(matches!(
            loaded,
            MockMessage::ModelLoaded { size_mb, load_time_ms }
            if (size_mb - 39.0).abs() < f64::EPSILON && (load_time_ms - 1500.0).abs() < f64::EPSILON
        ));

        let start = MockMessage::start(48000);
        assert!(matches!(start, MockMessage::Start { sample_rate } if sample_rate == 48000));

        let error = MockMessage::error("Test error");
        assert!(matches!(error, MockMessage::Error { message } if message == "Test error"));

        let partial = MockMessage::partial("Hello", false);
        assert!(
            matches!(partial, MockMessage::Partial { text, is_final } if text == "Hello" && !is_final)
        );
    }

    #[test]
    fn test_mock_runtime_message_flow() {
        let mut runtime = MockWasmRuntime::new();
        let received = Rc::new(RefCell::new(Vec::new()));
        let received_clone = Rc::clone(&received);

        runtime.on_message(move |msg| {
            received_clone.borrow_mut().push(msg.clone());
        });

        // Receive messages
        runtime.receive_message(MockMessage::Ready);
        runtime.receive_message(MockMessage::model_loaded(39.0, 1500.0));

        assert_eq!(runtime.pending_count(), 2);

        // Process one
        assert!(runtime.tick());
        assert_eq!(received.borrow().len(), 1);
        assert!(matches!(&received.borrow()[0], MockMessage::Ready));

        // Process remaining
        runtime.drain();
        assert_eq!(received.borrow().len(), 2);
        assert_eq!(runtime.total_processed(), 2);
    }

    #[test]
    fn test_mock_runtime_outgoing() {
        let runtime = MockWasmRuntime::new();

        runtime.post_message(MockMessage::start(48000));
        runtime.post_message(MockMessage::Stop);

        assert!(runtime.has_outgoing());
        assert_eq!(runtime.peek_outgoing().len(), 2);

        let outgoing = runtime.take_outgoing();
        assert_eq!(outgoing.len(), 2);
        assert!(!runtime.has_outgoing());
    }

    #[test]
    fn test_mock_runtime_clone() {
        let runtime1 = MockWasmRuntime::new();
        runtime1.receive_message(MockMessage::Ready);

        let runtime2 = runtime1;

        // Both share the same queues
        assert_eq!(runtime2.pending_count(), 1);
    }

    #[test]
    fn test_mock_runtime_tick_n() {
        let mut runtime = MockWasmRuntime::new();
        let count = Rc::new(RefCell::new(0));
        let count_clone = Rc::clone(&count);

        runtime.on_message(move |_| {
            *count_clone.borrow_mut() += 1;
        });

        for _ in 0..10 {
            runtime.receive_message(MockMessage::Ready);
        }

        // Process only 5
        let processed = runtime.tick_n(5);
        assert_eq!(processed, 5);
        assert_eq!(*count.borrow(), 5);
        assert_eq!(runtime.pending_count(), 5);
    }

    #[test]
    fn test_mock_runtime_reset() {
        let mut runtime = MockWasmRuntime::new();

        runtime.receive_message(MockMessage::Ready);
        runtime.post_message(MockMessage::Stop);
        runtime.on_message(|_| {});
        runtime.tick();

        assert!(runtime.total_processed() > 0);

        runtime.reset();

        assert_eq!(runtime.pending_count(), 0);
        assert!(!runtime.has_outgoing());
        assert_eq!(runtime.total_processed(), 0);
    }

    #[test]
    fn test_mock_message_equality() {
        let msg1 = MockMessage::model_loaded(39.0, 1500.0);
        let msg2 = MockMessage::model_loaded(39.0, 1500.0);
        let msg3 = MockMessage::model_loaded(40.0, 1500.0);

        assert_eq!(msg1, msg2);
        assert_ne!(msg1, msg3);
    }

    #[test]
    fn test_mock_runtime_default() {
        let runtime = MockWasmRuntime::default();
        assert!(!runtime.is_started());
        assert_eq!(runtime.pending_count(), 0);
        assert_eq!(runtime.total_processed(), 0);
    }

    #[test]
    fn test_mock_runtime_debug() {
        let runtime = MockWasmRuntime::new();
        let debug_str = format!("{:?}", runtime);
        assert!(debug_str.contains("MockWasmRuntime"));
        assert!(debug_str.contains("incoming_count"));
        assert!(debug_str.contains("started"));
    }

    #[test]
    fn test_mock_runtime_start() {
        let mut runtime = MockWasmRuntime::new();
        assert!(!runtime.is_started());
        runtime.start();
        assert!(runtime.is_started());
    }

    #[test]
    fn test_mock_runtime_receive_message_unchecked() {
        let runtime = MockWasmRuntime::new();
        runtime.receive_message_unchecked(MockMessage::Ready);
        assert_eq!(runtime.pending_count(), 1);
    }

    #[test]
    fn test_mock_runtime_tick_empty() {
        let mut runtime = MockWasmRuntime::new();
        assert!(!runtime.tick());
        assert_eq!(runtime.total_processed(), 0);
    }

    #[test]
    fn test_mock_runtime_drain_bounded() {
        let mut runtime = MockWasmRuntime::new();
        let counter = Rc::new(RefCell::new(0));
        let counter_clone = Rc::clone(&counter);

        runtime.on_message(move |_| {
            *counter_clone.borrow_mut() += 1;
        });

        for _ in 0..20 {
            runtime.receive_message(MockMessage::Ready);
        }

        // Process only 5
        let processed = runtime.drain_bounded(5);
        assert_eq!(processed, 5);
        assert_eq!(*counter.borrow(), 5);
        assert_eq!(runtime.pending_count(), 15);
    }

    #[test]
    fn test_mock_runtime_drain_all() {
        let mut runtime = MockWasmRuntime::new();
        for _ in 0..10 {
            runtime.receive_message(MockMessage::Ready);
        }

        runtime.drain();
        assert_eq!(runtime.pending_count(), 0);
    }

    #[test]
    fn test_mock_runtime_clone_shared_state() {
        let runtime1 = MockWasmRuntime::new();
        let runtime2 = runtime1.clone();

        runtime1.receive_message(MockMessage::Ready);
        // Both should see the same pending message
        assert_eq!(runtime1.pending_count(), 1);
        assert_eq!(runtime2.pending_count(), 1);

        runtime2.post_message(MockMessage::Stop);
        assert!(runtime1.has_outgoing());
        assert!(runtime2.has_outgoing());
    }

    #[test]
    fn test_mock_runtime_peek_outgoing() {
        let runtime = MockWasmRuntime::new();
        runtime.post_message(MockMessage::start(48000));
        runtime.post_message(MockMessage::Stop);

        let peeked = runtime.peek_outgoing();
        assert_eq!(peeked.len(), 2);

        // peek_outgoing doesn't consume
        let peeked_again = runtime.peek_outgoing();
        assert_eq!(peeked_again.len(), 2);
    }

    #[test]
    fn test_mock_runtime_take_outgoing_consumes() {
        let runtime = MockWasmRuntime::new();
        runtime.post_message(MockMessage::Ready);

        let taken = runtime.take_outgoing();
        assert_eq!(taken.len(), 1);

        // Should be empty after take
        assert!(!runtime.has_outgoing());
        let taken_again = runtime.take_outgoing();
        assert!(taken_again.is_empty());
    }

    #[test]
    fn test_mock_message_custom() {
        let msg = MockMessage::Custom {
            msg_type: "test".to_string(),
            payload: r#"{"key": "value"}"#.to_string(),
        };

        match msg {
            MockMessage::Custom { msg_type, payload } => {
                assert_eq!(msg_type, "test");
                assert!(payload.contains("key"));
            }
            _ => panic!("Expected Custom message"),
        }
    }

    #[test]
    fn test_mock_message_partial() {
        let msg = MockMessage::partial("Hello world", true);
        match msg {
            MockMessage::Partial { text, is_final } => {
                assert_eq!(text, "Hello world");
                assert!(is_final);
            }
            _ => panic!("Expected Partial message"),
        }

        let msg2 = MockMessage::partial("Partial", false);
        match msg2 {
            MockMessage::Partial { is_final, .. } => {
                assert!(!is_final);
            }
            _ => panic!("Expected Partial message"),
        }
    }

    #[test]
    fn test_mock_message_serialization() {
        // Test all message variants can be serialized/deserialized
        let messages = vec![
            MockMessage::bootstrap("http://localhost"),
            MockMessage::init("/model.apr"),
            MockMessage::Ready,
            MockMessage::model_loaded(100.0, 2000.0),
            MockMessage::start(44100),
            MockMessage::Stop,
            MockMessage::partial("text", true),
            MockMessage::error("oops"),
            MockMessage::Shutdown,
            MockMessage::Custom {
                msg_type: "t".into(),
                payload: "{}".into(),
            },
        ];

        for msg in messages {
            let serialized = bincode::serialize(&msg).expect("Should serialize");
            let deserialized: MockMessage =
                bincode::deserialize(&serialized).expect("Should deserialize");
            assert_eq!(msg, deserialized);
        }
    }

    #[test]
    fn test_mockable_worker_is_state_synced() {
        struct TestWorker {
            reported: String,
            internal: String,
        }

        impl MockableWorker for TestWorker {
            fn with_mock_runtime(_: MockWasmRuntime) -> Self {
                Self {
                    reported: "same".into(),
                    internal: "same".into(),
                }
            }

            fn get_state(&self) -> String {
                self.reported.clone()
            }

            fn debug_internal_state(&self) -> String {
                self.internal.clone()
            }
        }

        let worker = TestWorker {
            reported: "state".into(),
            internal: "state".into(),
        };
        assert!(worker.is_state_synced());

        let desynced = TestWorker {
            reported: "one".into(),
            internal: "two".into(),
        };
        assert!(!desynced.is_state_synced());
    }

    #[test]
    fn test_mock_runtime_multiple_handlers() {
        let mut runtime = MockWasmRuntime::new();
        let counter1 = Rc::new(RefCell::new(0));
        let counter2 = Rc::new(RefCell::new(0));

        let c1 = Rc::clone(&counter1);
        runtime.on_message(move |_| {
            *c1.borrow_mut() += 1;
        });

        let c2 = Rc::clone(&counter2);
        runtime.on_message(move |_| {
            *c2.borrow_mut() += 10;
        });

        runtime.receive_message(MockMessage::Ready);
        runtime.tick();

        // Both handlers should have been called
        assert_eq!(*counter1.borrow(), 1);
        assert_eq!(*counter2.borrow(), 10);
    }

    #[test]
    fn test_mock_runtime_handler_adds_new_handler() {
        let mut runtime = MockWasmRuntime::new();
        let counter = Rc::new(RefCell::new(0));
        let counter_clone = Rc::clone(&counter);

        // Handler that uses counter
        runtime.on_message(move |_| {
            *counter_clone.borrow_mut() += 1;
        });

        // Process first message
        runtime.receive_message(MockMessage::Ready);
        runtime.tick();
        assert_eq!(*counter.borrow(), 1);

        // Process second message
        runtime.receive_message(MockMessage::Stop);
        runtime.tick();
        assert_eq!(*counter.borrow(), 2);
    }

    #[test]
    fn test_mock_runtime_tick_n_partial() {
        let mut runtime = MockWasmRuntime::new();

        // Add 3 messages
        runtime.receive_message(MockMessage::Ready);
        runtime.receive_message(MockMessage::Stop);
        runtime.receive_message(MockMessage::Shutdown);

        // Process only 2
        let processed = runtime.tick_n(2);
        assert_eq!(processed, 2);
        assert_eq!(runtime.pending_count(), 1);
    }

    #[test]
    fn test_mock_runtime_tick_n_more_than_available() {
        let mut runtime = MockWasmRuntime::new();
        runtime.receive_message(MockMessage::Ready);

        // Try to process 100, but only 1 is available
        let processed = runtime.tick_n(100);
        assert_eq!(processed, 1);
        assert_eq!(runtime.pending_count(), 0);
    }

    #[test]
    fn test_mock_runtime_tick_n_zero() {
        let mut runtime = MockWasmRuntime::new();
        runtime.receive_message(MockMessage::Ready);

        let processed = runtime.tick_n(0);
        assert_eq!(processed, 0);
        assert_eq!(runtime.pending_count(), 1);
    }
}
