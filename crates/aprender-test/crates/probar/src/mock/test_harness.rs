//! WASM Callback Test Harness
//!
//! Per `PROBAR-SPEC-WASM-001` Section 2.2, this provides a test harness
//! for `WorkerManager`-style components that use callbacks.
//!
//! ## Iron Lotus Philosophy
//!
//! This harness tests ACTUAL code, not models. It would have caught
//! the WAPR-QA-REGRESSION-005 state sync bug because it verifies that
//! `get_state()` returns the correct value after callback processing.

use super::wasm_runtime::{MockMessage, MockWasmRuntime, MockableWorker};
use std::fmt::Debug;

/// A single test step with expected state
#[derive(Debug, Clone)]
pub struct TestStep {
    /// Message to send
    pub message: MockMessage,
    /// Expected state after processing
    pub expected_state: String,
    /// Optional description
    pub description: Option<String>,
}

impl TestStep {
    /// Create a new test step
    #[must_use]
    pub fn new(message: MockMessage, expected_state: &str) -> Self {
        Self {
            message,
            expected_state: expected_state.to_string(),
            description: None,
        }
    }

    /// Add a description to this step
    #[must_use]
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }
}

/// Assertion about component state
#[derive(Debug, Clone)]
pub enum StateAssertion {
    /// State equals expected value
    Equals(String),
    /// State contains substring
    Contains(String),
    /// State matches one of several values
    OneOf(Vec<String>),
    /// Custom predicate (as description)
    Custom(String),
}

impl StateAssertion {
    /// Check if state satisfies the assertion
    #[must_use]
    pub fn check(&self, actual: &str) -> bool {
        match self {
            Self::Equals(expected) => actual == expected,
            Self::Contains(substring) => actual.contains(substring),
            Self::OneOf(options) => options.iter().any(|o| actual == o),
            Self::Custom(_) => true, // Custom predicates need external evaluation
        }
    }

    /// Get a human-readable description of the assertion
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Equals(expected) => format!("state == \"{expected}\""),
            Self::Contains(substring) => format!("state contains \"{substring}\""),
            Self::OneOf(options) => format!("state in {:?}", options),
            Self::Custom(desc) => desc.clone(),
        }
    }
}

/// Test harness for WASM callback components
///
/// Wraps a component with mock runtime and provides testing utilities.
///
/// # Example
///
/// ```rust,ignore
/// let harness = WasmCallbackTestHarness::<MyWorker>::new();
///
/// // Spawn and verify initial state
/// harness.worker.spawn("model.apr").unwrap();
/// harness.assert_state("spawning");
///
/// // Simulate worker ready
/// harness.worker_ready();
/// harness.assert_state("loading");  // Would FAIL with state sync bug!
/// ```
pub struct WasmCallbackTestHarness<W: MockableWorker> {
    /// The worker component under test
    pub worker: W,
    /// The mock runtime (shared with worker)
    pub runtime: MockWasmRuntime,
    /// Test steps executed
    steps_executed: usize,
    /// Errors encountered
    errors: Vec<String>,
}

impl<W: MockableWorker> WasmCallbackTestHarness<W> {
    /// Create a new test harness
    #[must_use]
    pub fn new() -> Self {
        let runtime = MockWasmRuntime::new();
        let worker = W::with_mock_runtime(runtime.clone());
        Self {
            worker,
            runtime,
            steps_executed: 0,
            errors: Vec::new(),
        }
    }

    /// Get the current state
    #[must_use]
    pub fn state(&self) -> String {
        self.worker.get_state()
    }

    /// Assert that state equals expected value
    ///
    /// # Panics
    ///
    /// Panics if state doesn't match expected.
    pub fn assert_state(&self, expected: &str) {
        let actual = self.worker.get_state();
        assert_eq!(
            actual, expected,
            "State mismatch: expected '{}', got '{}'",
            expected, actual
        );
    }

    /// Assert that state satisfies a predicate
    ///
    /// # Panics
    ///
    /// Panics if assertion fails.
    pub fn assert(&self, assertion: &StateAssertion) {
        let actual = self.worker.get_state();
        assert!(
            assertion.check(&actual),
            "Assertion failed: {} (actual: '{}')",
            assertion.describe(),
            actual
        );
    }

    /// Check for state synchronization (catches WAPR-QA-REGRESSION-005 type bugs)
    ///
    /// # Panics
    ///
    /// Panics if internal state differs from reported state.
    pub fn assert_state_synced(&self) {
        let reported = self.worker.get_state();
        let internal = self.worker.debug_internal_state();
        assert_eq!(
            reported, internal,
            "STATE DESYNC DETECTED! Reported: '{}', Internal: '{}'\n\
             This indicates a bug like WAPR-QA-REGRESSION-005 where closure \
             updates a different variable than state checks use.",
            reported, internal
        );
    }

    /// Simulate worker becoming ready
    pub fn worker_ready(&mut self) {
        self.runtime.receive_message(MockMessage::Ready);
        self.runtime.tick();
        self.steps_executed += 1;
    }

    /// Simulate model loaded
    pub fn model_loaded(&mut self, size_mb: f64, load_time_ms: f64) {
        self.runtime.receive_message(MockMessage::ModelLoaded {
            size_mb,
            load_time_ms,
        });
        self.runtime.tick();
        self.steps_executed += 1;
    }

    /// Simulate an error
    pub fn worker_error(&mut self, message: &str) {
        self.runtime.receive_message(MockMessage::Error {
            message: message.to_string(),
        });
        self.runtime.tick();
        self.steps_executed += 1;
    }

    /// Send a custom message and tick
    pub fn send_message(&mut self, msg: MockMessage) {
        self.runtime.receive_message(msg);
        self.runtime.tick();
        self.steps_executed += 1;
    }

    /// Execute a sequence of test steps
    ///
    /// # Errors
    ///
    /// Returns error if any step's expected state doesn't match.
    pub fn execute_steps(&mut self, steps: &[TestStep]) -> Result<(), String> {
        for (i, step) in steps.iter().enumerate() {
            self.runtime.receive_message(step.message.clone());
            self.runtime.tick();
            self.steps_executed += 1;

            let actual = self.worker.get_state();
            if actual != step.expected_state {
                let desc = step
                    .description
                    .as_ref()
                    .map(|d| format!(" ({})", d))
                    .unwrap_or_default();
                return Err(format!(
                    "Step {}{}: expected state '{}', got '{}'",
                    i + 1,
                    desc,
                    step.expected_state,
                    actual
                ));
            }
        }
        Ok(())
    }

    /// Execute steps and collect all errors (don't fail fast)
    pub fn execute_steps_all(&mut self, steps: &[TestStep]) -> Vec<String> {
        let mut errors = Vec::new();

        for (i, step) in steps.iter().enumerate() {
            self.runtime.receive_message(step.message.clone());
            self.runtime.tick();
            self.steps_executed += 1;

            let actual = self.worker.get_state();
            if actual != step.expected_state {
                let desc = step
                    .description
                    .as_ref()
                    .map(|d| format!(" ({})", d))
                    .unwrap_or_default();
                errors.push(format!(
                    "Step {}{}: expected state '{}', got '{}'",
                    i + 1,
                    desc,
                    step.expected_state,
                    actual
                ));
            }
        }

        errors
    }

    /// Get the happy path test steps for a typical worker lifecycle
    #[must_use]
    pub fn happy_path_steps() -> Vec<TestStep> {
        vec![
            TestStep::new(MockMessage::Ready, "loading").with_description("Worker ready"),
            TestStep::new(MockMessage::model_loaded(39.0, 1500.0), "ready")
                .with_description("Model loaded"),
            TestStep::new(MockMessage::start(48000), "recording")
                .with_description("Recording started"),
            TestStep::new(MockMessage::Stop, "ready").with_description("Recording stopped"),
        ]
    }

    /// Get steps executed count
    #[must_use]
    pub fn steps_executed(&self) -> usize {
        self.steps_executed
    }

    /// Get recorded errors
    #[must_use]
    pub fn errors(&self) -> &[String] {
        &self.errors
    }

    /// Check if harness has errors
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Process all pending messages
    pub fn drain(&mut self) {
        self.runtime.drain();
    }

    /// Get pending message count
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.runtime.pending_count()
    }
}

impl<W: MockableWorker> Default for WasmCallbackTestHarness<W> {
    fn default() -> Self {
        Self::new()
    }
}

impl<W: MockableWorker> std::fmt::Debug for WasmCallbackTestHarness<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WasmCallbackTestHarness")
            .field("worker_state", &self.worker.get_state())
            .field("runtime", &self.runtime)
            .field("steps_executed", &self.steps_executed)
            .field("errors_count", &self.errors.len())
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // Simple mock worker for testing the harness itself
    struct SimpleWorker {
        state: String,
        #[allow(dead_code)]
        runtime: MockWasmRuntime,
    }

    impl MockableWorker for SimpleWorker {
        fn with_mock_runtime(mut runtime: MockWasmRuntime) -> Self {
            let worker = Self {
                state: "uninitialized".to_string(),
                runtime: runtime.clone(),
            };

            // Set up message handler that updates state
            let state_ptr = std::rc::Rc::new(std::cell::RefCell::new("uninitialized".to_string()));
            let state_clone = std::rc::Rc::clone(&state_ptr);

            runtime.on_message(move |msg| {
                let new_state = match msg {
                    MockMessage::Ready => "loading",
                    MockMessage::ModelLoaded { .. } => "ready",
                    MockMessage::Start { .. } => "recording",
                    MockMessage::Stop => "ready",
                    MockMessage::Error { .. } => "error",
                    MockMessage::Shutdown => "shutdown",
                    _ => return,
                };
                *state_clone.borrow_mut() = new_state.to_string();
            });

            // HACK: This is a simplified test implementation
            // In real code, the state would be properly shared
            worker
        }

        fn get_state(&self) -> String {
            self.state.clone()
        }
    }

    #[test]
    fn test_test_step_creation() {
        let step = TestStep::new(MockMessage::Ready, "loading").with_description("Worker ready");

        assert!(matches!(step.message, MockMessage::Ready));
        assert_eq!(step.expected_state, "loading");
        assert_eq!(step.description, Some("Worker ready".to_string()));
    }

    #[test]
    fn test_state_assertion_equals() {
        let assertion = StateAssertion::Equals("ready".to_string());
        assert!(assertion.check("ready"));
        assert!(!assertion.check("loading"));
    }

    #[test]
    fn test_state_assertion_contains() {
        let assertion = StateAssertion::Contains("load".to_string());
        assert!(assertion.check("loading"));
        assert!(assertion.check("loaded"));
        assert!(!assertion.check("ready"));
    }

    #[test]
    fn test_state_assertion_one_of() {
        let assertion = StateAssertion::OneOf(vec!["ready".to_string(), "loading".to_string()]);
        assert!(assertion.check("ready"));
        assert!(assertion.check("loading"));
        assert!(!assertion.check("error"));
    }

    #[test]
    fn test_state_assertion_describe() {
        assert_eq!(
            StateAssertion::Equals("ready".to_string()).describe(),
            r#"state == "ready""#
        );
        assert_eq!(
            StateAssertion::Contains("load".to_string()).describe(),
            r#"state contains "load""#
        );
    }

    #[test]
    fn test_harness_happy_path_steps() {
        let steps = WasmCallbackTestHarness::<SimpleWorker>::happy_path_steps();
        assert!(!steps.is_empty());
        assert!(matches!(steps[0].message, MockMessage::Ready));
    }

    #[test]
    fn test_state_assertion_custom() {
        let assertion = StateAssertion::Custom("custom check".to_string());
        // Custom assertions always return true (need external evaluation)
        assert!(assertion.check("anything"));
        assert_eq!(assertion.describe(), "custom check");
    }

    #[test]
    fn test_state_assertion_one_of_describe() {
        let assertion = StateAssertion::OneOf(vec!["ready".to_string(), "loading".to_string()]);
        let desc = assertion.describe();
        assert!(desc.contains("ready"));
        assert!(desc.contains("loading"));
    }

    // Stateful mock worker that actually updates state
    struct StatefulWorker {
        state: std::rc::Rc<std::cell::RefCell<String>>,
        #[allow(dead_code)]
        runtime: MockWasmRuntime,
    }

    impl MockableWorker for StatefulWorker {
        fn with_mock_runtime(mut runtime: MockWasmRuntime) -> Self {
            let state_ptr = std::rc::Rc::new(std::cell::RefCell::new("uninitialized".to_string()));
            let state_clone = std::rc::Rc::clone(&state_ptr);

            runtime.on_message(move |msg| {
                let new_state = match msg {
                    MockMessage::Ready => "loading",
                    MockMessage::ModelLoaded { .. } => "ready",
                    MockMessage::Start { .. } => "recording",
                    MockMessage::Stop => "ready",
                    MockMessage::Error { .. } => "error",
                    MockMessage::Shutdown => "shutdown",
                    _ => return,
                };
                *state_clone.borrow_mut() = new_state.to_string();
            });

            Self {
                state: state_ptr,
                runtime,
            }
        }

        fn get_state(&self) -> String {
            self.state.borrow().clone()
        }

        fn debug_internal_state(&self) -> String {
            self.state.borrow().clone()
        }
    }

    #[test]
    fn test_harness_new() {
        let harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        assert_eq!(harness.steps_executed(), 0);
        assert!(!harness.has_errors());
        assert!(harness.errors().is_empty());
    }

    #[test]
    fn test_harness_worker_ready() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        harness.worker_ready();
        assert_eq!(harness.steps_executed(), 1);
        assert_eq!(harness.worker.get_state(), "loading");
    }

    #[test]
    fn test_harness_model_loaded() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        harness.worker_ready();
        harness.model_loaded(39.0, 1500.0);
        assert_eq!(harness.steps_executed(), 2);
        assert_eq!(harness.worker.get_state(), "ready");
    }

    #[test]
    fn test_harness_worker_error() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        harness.worker_ready();
        harness.worker_error("test error");
        assert_eq!(harness.worker.get_state(), "error");
    }

    #[test]
    fn test_harness_send_message() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        harness.send_message(MockMessage::Shutdown);
        assert_eq!(harness.worker.get_state(), "shutdown");
    }

    #[test]
    fn test_harness_assert_state() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        harness.worker_ready();
        harness.assert_state("loading");
    }

    #[test]
    fn test_harness_assert_predicate() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        harness.worker_ready();
        harness.assert(&StateAssertion::Equals("loading".to_string()));
        harness.assert(&StateAssertion::Contains("load".to_string()));
    }

    #[test]
    fn test_harness_assert_state_synced() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        harness.worker_ready();
        harness.assert_state_synced(); // Should not panic
    }

    #[test]
    fn test_harness_execute_steps_success() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        let steps = vec![
            TestStep::new(MockMessage::Ready, "loading"),
            TestStep::new(MockMessage::model_loaded(39.0, 1500.0), "ready"),
        ];
        let result = harness.execute_steps(&steps);
        assert!(result.is_ok());
        assert_eq!(harness.steps_executed(), 2);
    }

    #[test]
    fn test_harness_execute_steps_failure() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        let steps = vec![TestStep::new(MockMessage::Ready, "wrong_state")];
        let result = harness.execute_steps(&steps);
        assert!(result.is_err());
    }

    #[test]
    fn test_harness_execute_steps_failure_with_description() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        let steps =
            vec![TestStep::new(MockMessage::Ready, "wrong_state").with_description("Worker ready")];
        let result = harness.execute_steps(&steps);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Worker ready"));
    }

    #[test]
    fn test_harness_execute_steps_all() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        let steps = vec![
            TestStep::new(MockMessage::Ready, "wrong1"),
            TestStep::new(MockMessage::model_loaded(39.0, 1500.0), "wrong2"),
        ];
        let errors = harness.execute_steps_all(&steps);
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_harness_execute_steps_all_with_description() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        let steps = vec![TestStep::new(MockMessage::Ready, "wrong").with_description("Test step")];
        let errors = harness.execute_steps_all(&steps);
        assert!(!errors.is_empty());
        assert!(errors[0].contains("Test step"));
    }

    #[test]
    fn test_harness_default() {
        let harness: WasmCallbackTestHarness<StatefulWorker> = WasmCallbackTestHarness::default();
        assert_eq!(harness.steps_executed(), 0);
        assert!(!harness.has_errors());
    }

    #[test]
    fn test_harness_debug() {
        let harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        let debug_str = format!("{:?}", harness);
        assert!(debug_str.contains("WasmCallbackTestHarness"));
        assert!(debug_str.contains("steps_executed"));
    }

    #[test]
    fn test_harness_state() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        assert_eq!(harness.state(), "uninitialized");
        harness.worker_ready();
        assert_eq!(harness.state(), "loading");
    }

    #[test]
    fn test_harness_drain() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        harness.runtime.receive_message(MockMessage::Ready);
        harness
            .runtime
            .receive_message(MockMessage::model_loaded(39.0, 1500.0));
        assert_eq!(harness.pending_count(), 2);
        harness.drain();
        assert_eq!(harness.pending_count(), 0);
    }

    #[test]
    fn test_harness_pending_count() {
        let harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        assert_eq!(harness.pending_count(), 0);
        harness.runtime.receive_message(MockMessage::Ready);
        assert_eq!(harness.pending_count(), 1);
    }

    #[test]
    fn test_test_step_without_description() {
        let step = TestStep::new(MockMessage::Ready, "loading");
        assert!(step.description.is_none());
    }

    #[test]
    fn test_execute_steps_success_no_description() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        let steps = vec![
            TestStep::new(MockMessage::Ready, "loading"),
            TestStep::new(MockMessage::model_loaded(39.0, 1500.0), "ready"),
        ];
        let result = harness.execute_steps(&steps);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_steps_all_success() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        let steps = vec![TestStep::new(MockMessage::Ready, "loading")];
        let errors = harness.execute_steps_all(&steps);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_execute_steps_all_no_description() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        let steps = vec![TestStep::new(MockMessage::Ready, "wrong_state")];
        let errors = harness.execute_steps_all(&steps);
        assert!(!errors.is_empty());
        // Should contain step number but no description
        assert!(errors[0].contains("Step 1:"));
    }

    #[test]
    fn test_state_assertion_one_of_empty() {
        let assertion = StateAssertion::OneOf(vec![]);
        assert!(!assertion.check("any"));
    }

    #[test]
    fn test_harness_errors_initially_empty() {
        let harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        assert!(harness.errors().is_empty());
        assert!(!harness.has_errors());
    }

    #[test]
    fn test_harness_full_lifecycle() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();

        // Worker ready
        harness.worker_ready();
        harness.assert_state("loading");

        // Model loaded
        harness.model_loaded(39.0, 1500.0);
        harness.assert_state("ready");

        // Start recording
        harness.send_message(MockMessage::start(48000));
        harness.assert_state("recording");

        // Stop recording
        harness.send_message(MockMessage::Stop);
        harness.assert_state("ready");

        assert_eq!(harness.steps_executed(), 4);
    }

    #[test]
    fn test_harness_shutdown() {
        let mut harness = WasmCallbackTestHarness::<StatefulWorker>::new();
        harness.send_message(MockMessage::Shutdown);
        assert_eq!(harness.state(), "shutdown");
    }

    #[test]
    fn test_state_assertion_equals_empty() {
        let assertion = StateAssertion::Equals(String::new());
        assert!(assertion.check(""));
        assert!(!assertion.check("something"));
    }

    #[test]
    fn test_state_assertion_contains_empty() {
        let assertion = StateAssertion::Contains(String::new());
        // Empty string is contained in any string
        assert!(assertion.check("anything"));
        assert!(assertion.check(""));
    }

    #[test]
    fn test_happy_path_steps_structure() {
        let steps = WasmCallbackTestHarness::<StatefulWorker>::happy_path_steps();
        assert_eq!(steps.len(), 4);

        // Check all steps have descriptions
        for step in &steps {
            assert!(step.description.is_some());
        }
    }
}
