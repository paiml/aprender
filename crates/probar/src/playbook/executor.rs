//! Playbook execution engine with transition assertions.
//!
//! Executes state machine transitions and verifies assertions.
//! Tracks timing for complexity analysis.

use super::complexity::{check_complexity_violation, ComplexityResult};
use super::schema::{Action, Assertion, Playbook, Transition, WaitCondition};
use std::time::{Duration, Instant};

/// Result of executing a playbook.
#[derive(Debug)]
pub struct ExecutionResult {
    /// Whether the playbook completed successfully
    pub success: bool,
    /// Current state after execution
    pub final_state: String,
    /// Executed transitions
    pub transitions_executed: Vec<TransitionResult>,
    /// Total execution time
    pub total_time: Duration,
    /// Performance metrics
    pub metrics: ExecutionMetrics,
    /// Assertion failures
    pub assertion_failures: Vec<AssertionFailure>,
    /// Complexity analysis (if performance budget specified)
    pub complexity_result: Option<ComplexityResult>,
}

/// Result of a single transition execution.
#[derive(Debug, Clone)]
pub struct TransitionResult {
    /// Transition ID
    pub transition_id: String,
    /// Source state
    pub from_state: String,
    /// Target state
    pub to_state: String,
    /// Execution time for this transition
    pub duration: Duration,
    /// Whether assertions passed
    pub assertions_passed: bool,
    /// Individual assertion results
    pub assertion_results: Vec<AssertionResult>,
}

/// Result of a single assertion check.
#[derive(Debug, Clone)]
pub struct AssertionResult {
    /// Whether the assertion passed
    pub passed: bool,
    /// Assertion description
    pub description: String,
    /// Error message if failed
    pub error: Option<String>,
}

/// Details of an assertion failure.
#[derive(Debug, Clone)]
pub struct AssertionFailure {
    /// Transition where failure occurred
    pub transition_id: String,
    /// Assertion that failed
    pub assertion_description: String,
    /// Error details
    pub error: String,
}

/// Execution performance metrics.
#[derive(Debug, Clone, Default)]
pub struct ExecutionMetrics {
    /// Time per transition (for complexity analysis)
    pub transition_times: Vec<(usize, f64)>,
    /// Peak memory usage (if tracked)
    pub peak_memory_bytes: Option<u64>,
    /// Total transitions executed
    pub transition_count: usize,
}

/// Trait for executing actions in a playbook.
/// Implement this for your specific testing environment.
pub trait ActionExecutor {
    /// Execute a click action.
    fn click(&mut self, selector: &str) -> Result<(), ExecutorError>;

    /// Execute a type action.
    fn type_text(&mut self, selector: &str, text: &str) -> Result<(), ExecutorError>;

    /// Execute a wait action.
    fn wait(&mut self, condition: &WaitCondition) -> Result<(), ExecutorError>;

    /// Execute a navigation action.
    fn navigate(&mut self, url: &str) -> Result<(), ExecutorError>;

    /// Execute a script action.
    fn execute_script(&mut self, code: &str) -> Result<String, ExecutorError>;

    /// Take a screenshot.
    fn screenshot(&mut self, name: &str) -> Result<(), ExecutorError>;

    /// Check if element exists.
    fn element_exists(&self, selector: &str) -> Result<bool, ExecutorError>;

    /// Get element text.
    fn get_text(&self, selector: &str) -> Result<String, ExecutorError>;

    /// Get element attribute.
    fn get_attribute(&self, selector: &str, attribute: &str) -> Result<String, ExecutorError>;

    /// Get current URL.
    fn get_url(&self) -> Result<String, ExecutorError>;

    /// Evaluate JavaScript expression.
    fn evaluate(&self, expression: &str) -> Result<bool, ExecutorError>;
}

/// Errors during playbook execution.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ExecutorError {
    #[error("Element not found: {selector}")]
    ElementNotFound { selector: String },

    #[error("Timeout waiting for condition")]
    Timeout,

    #[error("Navigation failed: {url}")]
    NavigationFailed { url: String },

    #[error("Script execution failed: {message}")]
    ScriptError { message: String },

    #[error("Assertion failed: {message}")]
    AssertionFailed { message: String },

    #[error("Invalid transition: no transition from '{state}' with event '{event}'")]
    InvalidTransition { state: String, event: String },

    #[error("Performance budget exceeded: {message}")]
    PerformanceBudgetExceeded { message: String },
}

/// Playbook execution engine.
pub struct PlaybookExecutor<E: ActionExecutor> {
    playbook: Playbook,
    executor: E,
    current_state: String,
    transition_count: usize,
}

impl<E: ActionExecutor> PlaybookExecutor<E> {
    /// Create a new executor for the given playbook.
    pub fn new(playbook: Playbook, executor: E) -> Self {
        let initial = playbook.machine.initial.clone();
        Self {
            playbook,
            executor,
            current_state: initial,
            transition_count: 0,
        }
    }

    /// Execute the playbook by following the given event sequence.
    pub fn execute(&mut self, events: &[&str]) -> ExecutionResult {
        let start = Instant::now();
        let mut transitions_executed = Vec::new();
        let mut assertion_failures = Vec::new();
        let mut metrics = ExecutionMetrics::default();
        let mut success = true;

        for event in events {
            match self.trigger_event(event) {
                Ok(result) => {
                    // Track timing for complexity analysis
                    metrics.transition_times.push((
                        self.transition_count,
                        result.duration.as_secs_f64() * 1000.0,
                    ));

                    // Check for assertion failures
                    if !result.assertions_passed {
                        for ar in &result.assertion_results {
                            if !ar.passed {
                                assertion_failures.push(AssertionFailure {
                                    transition_id: result.transition_id.clone(),
                                    assertion_description: ar.description.clone(),
                                    error: ar.error.clone().unwrap_or_default(),
                                });
                            }
                        }
                        success = false;
                    }

                    transitions_executed.push(result);
                }
                Err(e) => {
                    assertion_failures.push(AssertionFailure {
                        transition_id: format!("event:{}", event),
                        assertion_description: "Transition execution".to_string(),
                        error: e.to_string(),
                    });
                    success = false;
                    break;
                }
            }
        }

        metrics.transition_count = transitions_executed.len();

        // Check complexity if budget specified
        let complexity_result =
            self.playbook.performance.complexity_class.map(|expected| {
                check_complexity_violation(metrics.transition_times.clone(), expected)
            });

        // Check for complexity violation
        if let Some(ref cr) = complexity_result {
            if cr.is_violation {
                success = false;
            }
        }

        ExecutionResult {
            success,
            final_state: self.current_state.clone(),
            transitions_executed,
            total_time: start.elapsed(),
            metrics,
            assertion_failures,
            complexity_result,
        }
    }

    /// Trigger an event and execute the corresponding transition.
    fn trigger_event(&mut self, event: &str) -> Result<TransitionResult, ExecutorError> {
        // Find matching transition and clone necessary data to avoid borrow issues
        let transition = self.find_transition(event)?;
        let transition_id = transition.id.clone();
        let from_state = transition.from.clone();
        let to_state = transition.to.clone();
        let transition_actions = transition.actions.clone();
        let transition_assertions = transition.assertions.clone();

        let start = Instant::now();

        // Clone exit actions to avoid borrow issues
        let exit_actions = self
            .playbook
            .machine
            .states
            .get(&self.current_state)
            .map(|s| s.on_exit.clone())
            .unwrap_or_default();

        // Execute exit actions for current state
        for action in &exit_actions {
            self.execute_action(action)?;
        }

        // Execute transition actions
        for action in &transition_actions {
            self.execute_action(action)?;
        }

        // Update current state
        self.current_state = to_state.clone();
        self.transition_count += 1;

        // Clone entry actions to avoid borrow issues
        let entry_actions = self
            .playbook
            .machine
            .states
            .get(&self.current_state)
            .map(|s| s.on_entry.clone())
            .unwrap_or_default();

        // Execute entry actions for new state
        for action in &entry_actions {
            self.execute_action(action)?;
        }

        // Check assertions
        let assertion_results = self.check_assertions(&transition_assertions);
        let assertions_passed = assertion_results.iter().all(|r| r.passed);

        let duration = start.elapsed();

        Ok(TransitionResult {
            transition_id,
            from_state,
            to_state,
            duration,
            assertions_passed,
            assertion_results,
        })
    }

    /// Find a transition matching the current state and event.
    fn find_transition(&self, event: &str) -> Result<&Transition, ExecutorError> {
        self.playbook
            .machine
            .transitions
            .iter()
            .find(|t| t.from == self.current_state && t.event == event)
            .ok_or_else(|| ExecutorError::InvalidTransition {
                state: self.current_state.clone(),
                event: event.to_string(),
            })
    }

    /// Execute a single action.
    fn execute_action(&mut self, action: &Action) -> Result<(), ExecutorError> {
        match action {
            Action::Click { selector } => self.executor.click(selector),
            Action::Type { selector, text } => self.executor.type_text(selector, text),
            Action::Wait { condition } => self.executor.wait(condition),
            Action::Navigate { url } => self.executor.navigate(url),
            Action::Script { code } => self.executor.execute_script(code).map(|_| ()),
            Action::Screenshot { name } => self.executor.screenshot(name),
        }
    }

    /// Check all assertions and return results.
    fn check_assertions(&self, assertions: &[Assertion]) -> Vec<AssertionResult> {
        assertions
            .iter()
            .map(|assertion| self.check_assertion(assertion))
            .collect()
    }

    /// Check a single assertion.
    fn check_assertion(&self, assertion: &Assertion) -> AssertionResult {
        match assertion {
            Assertion::ElementExists { selector } => match self.executor.element_exists(selector) {
                Ok(true) => AssertionResult {
                    passed: true,
                    description: format!("Element exists: {}", selector),
                    error: None,
                },
                Ok(false) => AssertionResult {
                    passed: false,
                    description: format!("Element exists: {}", selector),
                    error: Some(format!("Element not found: {}", selector)),
                },
                Err(e) => AssertionResult {
                    passed: false,
                    description: format!("Element exists: {}", selector),
                    error: Some(e.to_string()),
                },
            },
            Assertion::TextEquals { selector, expected } => {
                match self.executor.get_text(selector) {
                    Ok(actual) if actual == *expected => AssertionResult {
                        passed: true,
                        description: format!("Text equals '{}': {}", expected, selector),
                        error: None,
                    },
                    Ok(actual) => AssertionResult {
                        passed: false,
                        description: format!("Text equals '{}': {}", expected, selector),
                        error: Some(format!("Expected '{}', got '{}'", expected, actual)),
                    },
                    Err(e) => AssertionResult {
                        passed: false,
                        description: format!("Text equals '{}': {}", expected, selector),
                        error: Some(e.to_string()),
                    },
                }
            }
            Assertion::TextContains {
                selector,
                substring,
            } => match self.executor.get_text(selector) {
                Ok(actual) if actual.contains(substring) => AssertionResult {
                    passed: true,
                    description: format!("Text contains '{}': {}", substring, selector),
                    error: None,
                },
                Ok(actual) => AssertionResult {
                    passed: false,
                    description: format!("Text contains '{}': {}", substring, selector),
                    error: Some(format!("'{}' not found in '{}'", substring, actual)),
                },
                Err(e) => AssertionResult {
                    passed: false,
                    description: format!("Text contains '{}': {}", substring, selector),
                    error: Some(e.to_string()),
                },
            },
            Assertion::AttributeEquals {
                selector,
                attribute,
                expected,
            } => match self.executor.get_attribute(selector, attribute) {
                Ok(actual) if actual == *expected => AssertionResult {
                    passed: true,
                    description: format!("{}[{}] = '{}'", selector, attribute, expected),
                    error: None,
                },
                Ok(actual) => AssertionResult {
                    passed: false,
                    description: format!("{}[{}] = '{}'", selector, attribute, expected),
                    error: Some(format!("Expected '{}', got '{}'", expected, actual)),
                },
                Err(e) => AssertionResult {
                    passed: false,
                    description: format!("{}[{}] = '{}'", selector, attribute, expected),
                    error: Some(e.to_string()),
                },
            },
            Assertion::UrlMatches { pattern } => match self.executor.get_url() {
                Ok(url) => {
                    let matches = url.contains(pattern);
                    AssertionResult {
                        passed: matches,
                        description: format!("URL matches '{}'", pattern),
                        error: if matches {
                            None
                        } else {
                            Some(format!("URL '{}' does not match '{}'", url, pattern))
                        },
                    }
                }
                Err(e) => AssertionResult {
                    passed: false,
                    description: format!("URL matches '{}'", pattern),
                    error: Some(e.to_string()),
                },
            },
            Assertion::Script { expression } => match self.executor.evaluate(expression) {
                Ok(true) => AssertionResult {
                    passed: true,
                    description: format!("Script: {}", expression),
                    error: None,
                },
                Ok(false) => AssertionResult {
                    passed: false,
                    description: format!("Script: {}", expression),
                    error: Some("Expression evaluated to false".to_string()),
                },
                Err(e) => AssertionResult {
                    passed: false,
                    description: format!("Script: {}", expression),
                    error: Some(e.to_string()),
                },
            },
        }
    }

    /// Get the current state.
    pub fn current_state(&self) -> &str {
        &self.current_state
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.current_state = self.playbook.machine.initial.clone();
        self.transition_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Mock executor for testing
    struct MockExecutor {
        elements: HashMap<String, String>,
        url: String,
    }

    impl MockExecutor {
        fn new() -> Self {
            let mut elements = HashMap::new();
            elements.insert("#welcome".to_string(), "Welcome, User!".to_string());
            elements.insert("#login-btn".to_string(), "Login".to_string());

            Self {
                elements,
                url: "http://localhost/".to_string(),
            }
        }
    }

    impl ActionExecutor for MockExecutor {
        fn click(&mut self, _selector: &str) -> Result<(), ExecutorError> {
            Ok(())
        }

        fn type_text(&mut self, _selector: &str, _text: &str) -> Result<(), ExecutorError> {
            Ok(())
        }

        fn wait(&mut self, _condition: &WaitCondition) -> Result<(), ExecutorError> {
            Ok(())
        }

        fn navigate(&mut self, url: &str) -> Result<(), ExecutorError> {
            self.url = url.to_string();
            Ok(())
        }

        fn execute_script(&mut self, _code: &str) -> Result<String, ExecutorError> {
            Ok("undefined".to_string())
        }

        fn screenshot(&mut self, _name: &str) -> Result<(), ExecutorError> {
            Ok(())
        }

        fn element_exists(&self, selector: &str) -> Result<bool, ExecutorError> {
            Ok(self.elements.contains_key(selector))
        }

        fn get_text(&self, selector: &str) -> Result<String, ExecutorError> {
            self.elements
                .get(selector)
                .cloned()
                .ok_or_else(|| ExecutorError::ElementNotFound {
                    selector: selector.to_string(),
                })
        }

        fn get_attribute(
            &self,
            _selector: &str,
            _attribute: &str,
        ) -> Result<String, ExecutorError> {
            Ok("value".to_string())
        }

        fn get_url(&self) -> Result<String, ExecutorError> {
            Ok(self.url.clone())
        }

        fn evaluate(&self, _expression: &str) -> Result<bool, ExecutorError> {
            Ok(true)
        }
    }

    #[test]
    fn test_execute_simple_playbook() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
      final_state: true
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
      assertions:
        - type: element_exists
          selector: "#welcome"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let executor = MockExecutor::new();
        let mut runner = PlaybookExecutor::new(playbook, executor);

        let result = runner.execute(&["go"]);

        assert!(result.success);
        assert_eq!(result.final_state, "end");
        assert_eq!(result.transitions_executed.len(), 1);
        assert!(result.transitions_executed[0].assertions_passed);
    }

    #[test]
    fn test_assertion_failure() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
      final_state: true
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
      assertions:
        - type: element_exists
          selector: "#nonexistent"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let executor = MockExecutor::new();
        let mut runner = PlaybookExecutor::new(playbook, executor);

        let result = runner.execute(&["go"]);

        assert!(!result.success);
        assert!(!result.assertion_failures.is_empty());
    }

    #[test]
    fn test_invalid_transition() {
        let yaml = r#"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
      final_state: true
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let executor = MockExecutor::new();
        let mut runner = PlaybookExecutor::new(playbook, executor);

        let result = runner.execute(&["invalid_event"]);

        assert!(!result.success);
        assert!(!result.assertion_failures.is_empty());
    }

    #[test]
    fn test_current_state_and_reset() {
        let yaml = r#"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let executor = MockExecutor::new();
        let mut runner = PlaybookExecutor::new(playbook, executor);

        assert_eq!(runner.current_state(), "start");

        runner.execute(&["go"]);
        assert_eq!(runner.current_state(), "end");

        runner.reset();
        assert_eq!(runner.current_state(), "start");
    }

    #[test]
    fn test_text_equals_assertion_pass() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
      assertions:
        - type: text_equals
          selector: "#welcome"
          expected: "Welcome, User!"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let executor = MockExecutor::new();
        let mut runner = PlaybookExecutor::new(playbook, executor);

        let result = runner.execute(&["go"]);
        assert!(result.success);
    }

    #[test]
    fn test_text_equals_assertion_fail() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
      assertions:
        - type: text_equals
          selector: "#welcome"
          expected: "Wrong Text"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let executor = MockExecutor::new();
        let mut runner = PlaybookExecutor::new(playbook, executor);

        let result = runner.execute(&["go"]);
        assert!(!result.success);
    }

    #[test]
    fn test_text_contains_assertion_pass() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
      assertions:
        - type: text_contains
          selector: "#welcome"
          substring: "Welcome"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let executor = MockExecutor::new();
        let mut runner = PlaybookExecutor::new(playbook, executor);

        let result = runner.execute(&["go"]);
        assert!(result.success);
    }

    #[test]
    fn test_text_contains_assertion_fail() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
      assertions:
        - type: text_contains
          selector: "#welcome"
          substring: "Goodbye"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let executor = MockExecutor::new();
        let mut runner = PlaybookExecutor::new(playbook, executor);

        let result = runner.execute(&["go"]);
        assert!(!result.success);
    }

    #[test]
    fn test_attribute_equals_assertion_pass() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
      assertions:
        - type: attribute_equals
          selector: "#welcome"
          attribute: "data-test"
          expected: "value"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let executor = MockExecutor::new();
        let mut runner = PlaybookExecutor::new(playbook, executor);

        let result = runner.execute(&["go"]);
        assert!(result.success);
    }

    #[test]
    fn test_attribute_equals_assertion_fail() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
      assertions:
        - type: attribute_equals
          selector: "#welcome"
          attribute: "data-test"
          expected: "wrong_value"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let executor = MockExecutor::new();
        let mut runner = PlaybookExecutor::new(playbook, executor);

        let result = runner.execute(&["go"]);
        assert!(!result.success);
    }

    #[test]
    fn test_url_matches_assertion_pass() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
      assertions:
        - type: url_matches
          pattern: "localhost"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let executor = MockExecutor::new();
        let mut runner = PlaybookExecutor::new(playbook, executor);

        let result = runner.execute(&["go"]);
        assert!(result.success);
    }

    #[test]
    fn test_url_matches_assertion_fail() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
      assertions:
        - type: url_matches
          pattern: "example.com"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let executor = MockExecutor::new();
        let mut runner = PlaybookExecutor::new(playbook, executor);

        let result = runner.execute(&["go"]);
        assert!(!result.success);
    }

    #[test]
    fn test_script_assertion_pass() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
      assertions:
        - type: script
          expression: "true"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let executor = MockExecutor::new();
        let mut runner = PlaybookExecutor::new(playbook, executor);

        let result = runner.execute(&["go"]);
        assert!(result.success);
    }

    #[test]
    fn test_transition_with_actions() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
      actions:
        - type: click
          selector: "#button"
        - type: type
          selector: "#input"
          text: "hello"
        - type: wait
          condition:
            type: duration
            ms: 100
        - type: navigate
          url: "http://example.com"
        - type: script
          code: "console.log('test')"
        - type: screenshot
          name: "test_screenshot"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let executor = MockExecutor::new();
        let mut runner = PlaybookExecutor::new(playbook, executor);

        let result = runner.execute(&["go"]);
        assert!(result.success);
    }

    #[test]
    fn test_entry_exit_actions() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
      on_exit:
        - type: click
          selector: "#exit-start"
    middle:
      id: "middle"
      on_entry:
        - type: click
          selector: "#enter-middle"
      on_exit:
        - type: click
          selector: "#exit-middle"
    end:
      id: "end"
      on_entry:
        - type: click
          selector: "#enter-end"
  transitions:
    - id: "t1"
      from: "start"
      to: "middle"
      event: "step1"
    - id: "t2"
      from: "middle"
      to: "end"
      event: "step2"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let executor = MockExecutor::new();
        let mut runner = PlaybookExecutor::new(playbook, executor);

        let result = runner.execute(&["step1", "step2"]);
        assert!(result.success);
        assert_eq!(result.transitions_executed.len(), 2);
        assert_eq!(result.final_state, "end");
    }

    #[test]
    fn test_multiple_transitions() {
        let yaml = r#"
version: "1.0"
machine:
  id: "test"
  initial: "a"
  states:
    a:
      id: "a"
    b:
      id: "b"
    c:
      id: "c"
  transitions:
    - id: "t1"
      from: "a"
      to: "b"
      event: "next"
    - id: "t2"
      from: "b"
      to: "c"
      event: "next"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let executor = MockExecutor::new();
        let mut runner = PlaybookExecutor::new(playbook, executor);

        let result = runner.execute(&["next", "next"]);
        assert!(result.success);
        assert_eq!(result.final_state, "c");
        assert_eq!(result.metrics.transition_count, 2);
    }

    #[test]
    fn test_text_assertion_element_not_found() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
      assertions:
        - type: text_equals
          selector: "#nonexistent"
          expected: "some text"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let executor = MockExecutor::new();
        let mut runner = PlaybookExecutor::new(playbook, executor);

        let result = runner.execute(&["go"]);
        assert!(!result.success);
    }

    #[test]
    fn test_text_contains_element_not_found() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
      assertions:
        - type: text_contains
          selector: "#nonexistent"
          substring: "text"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let executor = MockExecutor::new();
        let mut runner = PlaybookExecutor::new(playbook, executor);

        let result = runner.execute(&["go"]);
        assert!(!result.success);
    }
}
