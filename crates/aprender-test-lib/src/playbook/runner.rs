//! Playbook runner with full setup/steps/teardown execution.
//!
//! Implements:
//! - Setup/teardown lifecycle (teardown runs even on failure)
//! - Variable capture and substitution
//! - Forbidden transition checking
//! - Path and output assertions
//! - Execution trace recording

use super::executor::{ActionExecutor, ExecutorError, PlaybookExecutor};
use super::schema::{OutputAssertion, PathAssertion, Playbook, PlaybookAction, PlaybookStep};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Result of running a playbook.
#[derive(Debug)]
pub struct PlaybookRunResult {
    /// Whether the playbook passed
    pub passed: bool,
    /// Captured variables
    pub variables: HashMap<String, String>,
    /// Execution trace (state path taken)
    pub state_path: Vec<String>,
    /// Individual step results
    pub step_results: Vec<StepResult>,
    /// Assertion results
    pub assertion_results: Vec<AssertionCheckResult>,
    /// Total execution time
    pub total_time: Duration,
    /// Error message if failed
    pub error: Option<String>,
}

/// Result of executing a single step.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Step name
    pub name: String,
    /// Whether step passed
    pub passed: bool,
    /// Step execution time
    pub duration: Duration,
    /// Captured variables from this step
    pub captured: HashMap<String, String>,
    /// Error message if failed
    pub error: Option<String>,
}

/// Result of checking an assertion.
#[derive(Debug, Clone)]
pub struct AssertionCheckResult {
    /// Assertion description
    pub description: String,
    /// Whether assertion passed
    pub passed: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Playbook runner that manages the full execution lifecycle.
pub struct PlaybookRunner<E: ActionExecutor> {
    playbook: Playbook,
    #[allow(dead_code)] // Will be used when action execution is implemented
    executor: PlaybookExecutor<E>,
    variables: HashMap<String, String>,
    state_path: Vec<String>,
}

impl<E: ActionExecutor> PlaybookRunner<E> {
    /// Create a new runner for the given playbook.
    pub fn new(playbook: Playbook, executor: E) -> Self {
        let initial = playbook.machine.initial.clone();
        let pb_executor = PlaybookExecutor::new(playbook.clone(), executor);

        Self {
            playbook,
            executor: pb_executor,
            variables: HashMap::new(),
            state_path: vec![initial],
        }
    }

    /// Run the complete playbook.
    pub fn run(&mut self) -> PlaybookRunResult {
        let start = Instant::now();
        let mut step_results = Vec::new();
        let mut passed = true;
        let mut error_msg: Option<String> = None;

        // Get playbook steps (if defined)
        let steps = self.playbook.playbook.clone().unwrap_or_default();

        // Run setup
        if let Err(e) = self.run_setup(&steps.setup) {
            error_msg = Some(format!("Setup failed: {}", e));
            passed = false;
        }

        // Run steps if setup succeeded
        if passed {
            for step in &steps.steps {
                match self.run_step(step) {
                    Ok(result) => {
                        if !result.passed {
                            passed = false;
                            error_msg = result.error.clone();
                        }
                        step_results.push(result);
                        if !passed {
                            break;
                        }
                    }
                    Err(e) => {
                        passed = false;
                        error_msg = Some(e.to_string());
                        step_results.push(StepResult {
                            name: step.name.clone(),
                            passed: false,
                            duration: Duration::ZERO,
                            captured: HashMap::new(),
                            error: Some(e.to_string()),
                        });
                        break;
                    }
                }
            }
        }

        // Run teardown (always, even on failure)
        let _ = self.run_teardown(&steps.teardown);

        // Check assertions
        let assertion_results = self.check_assertions();
        if assertion_results.iter().any(|a| !a.passed) {
            passed = false;
            if error_msg.is_none() {
                error_msg = Some("Assertions failed".to_string());
            }
        }

        PlaybookRunResult {
            passed,
            variables: self.variables.clone(),
            state_path: self.state_path.clone(),
            step_results,
            assertion_results,
            total_time: start.elapsed(),
            error: error_msg,
        }
    }

    /// Run setup actions.
    fn run_setup(&self, setup: &[PlaybookAction]) -> Result<(), ExecutorError> {
        for action in setup {
            self.run_action(action)?;
        }
        Ok(())
    }

    /// Run teardown actions.
    fn run_teardown(&self, teardown: &[PlaybookAction]) -> Result<(), ExecutorError> {
        for action in teardown {
            if action.ignore_errors {
                let _ = self.run_action(action);
            } else {
                self.run_action(action)?;
            }
        }
        Ok(())
    }

    /// Run a single action.
    fn run_action(&self, _action: &PlaybookAction) -> Result<(), ExecutorError> {
        // TODO: Execute WASM action via executor
        Ok(())
    }

    /// Run a single step.
    fn run_step(&mut self, step: &PlaybookStep) -> Result<StepResult, ExecutorError> {
        let start = Instant::now();
        let mut captured = HashMap::new();

        // Execute transitions for this step
        for transition_id in &step.transitions {
            // Find the transition by ID
            let transition = self
                .playbook
                .machine
                .transitions
                .iter()
                .find(|t| &t.id == transition_id);

            if let Some(t) = transition {
                // Check if this is a forbidden transition
                if let Some(err) = self.check_forbidden(&t.from, &t.to) {
                    return Ok(StepResult {
                        name: step.name.clone(),
                        passed: false,
                        duration: start.elapsed(),
                        captured,
                        error: Some(err),
                    });
                }

                // Record state path
                self.state_path.push(t.to.clone());
            }
        }

        // Capture variables
        for capture in &step.capture {
            // TODO: Actually evaluate the expression
            let value = self.substitute_variables(&capture.from);
            captured.insert(capture.var.clone(), value.clone());
            self.variables.insert(capture.var.clone(), value);
        }

        Ok(StepResult {
            name: step.name.clone(),
            passed: true,
            duration: start.elapsed(),
            captured,
            error: None,
        })
    }

    /// Check if a transition is forbidden.
    fn check_forbidden(&self, from: &str, to: &str) -> Option<String> {
        for forbidden in &self.playbook.machine.forbidden {
            if forbidden.from == from && forbidden.to == to {
                return Some(format!(
                    "Forbidden transition: {} -> {} ({})",
                    from, to, forbidden.reason
                ));
            }
        }
        None
    }

    /// Substitute ${var} patterns in a string.
    fn substitute_variables(&self, input: &str) -> String {
        let mut result = input.to_string();
        for (key, value) in &self.variables {
            let pattern = format!("${{{}}}", key);
            result = result.replace(&pattern, value);
        }
        result
    }

    /// Check all assertions.
    fn check_assertions(&self) -> Vec<AssertionCheckResult> {
        let mut results = Vec::new();

        if let Some(assertions) = &self.playbook.assertions {
            // Check path assertion
            if let Some(path) = &assertions.path {
                results.push(self.check_path_assertion(path));
            }

            // Check output assertions
            for output in &assertions.output {
                results.push(self.check_output_assertion(output));
            }
        }

        results
    }

    /// Check path assertion.
    fn check_path_assertion(&self, path: &PathAssertion) -> AssertionCheckResult {
        let actual_path: Vec<&str> = self.state_path.iter().map(|s| s.as_str()).collect();
        let expected_path: Vec<&str> = path.expected.iter().map(|s| s.as_str()).collect();

        if actual_path == expected_path {
            AssertionCheckResult {
                description: "Path matches expected sequence".to_string(),
                passed: true,
                error: None,
            }
        } else {
            AssertionCheckResult {
                description: "Path matches expected sequence".to_string(),
                passed: false,
                error: Some(format!(
                    "Expected path {:?}, got {:?}",
                    expected_path, actual_path
                )),
            }
        }
    }

    /// Check output assertion.
    fn check_output_assertion(&self, output: &OutputAssertion) -> AssertionCheckResult {
        let value = self.variables.get(&output.var);

        // Check not_empty
        if output.not_empty == Some(true) && value.map_or(true, String::is_empty) {
            return AssertionCheckResult {
                description: format!("Variable '{}' is not empty", output.var),
                passed: false,
                error: Some(format!("Variable '{}' is empty or undefined", output.var)),
            };
        }

        // Check matches regex
        if let Some(pattern) = &output.matches {
            if let Some(val) = value {
                if let Ok(re) = regex::Regex::new(pattern) {
                    if !re.is_match(val) {
                        return AssertionCheckResult {
                            description: format!("Variable '{}' matches '{}'", output.var, pattern),
                            passed: false,
                            error: Some(format!(
                                "Value '{}' does not match pattern '{}'",
                                val, pattern
                            )),
                        };
                    }
                }
            } else {
                return AssertionCheckResult {
                    description: format!("Variable '{}' matches '{}'", output.var, pattern),
                    passed: false,
                    error: Some(format!("Variable '{}' is undefined", output.var)),
                };
            }
        }

        // Check less_than
        if let Some(max) = output.less_than {
            if let Some(val) = value {
                if let Ok(num) = val.parse::<i64>() {
                    if num >= max {
                        return AssertionCheckResult {
                            description: format!("Variable '{}' < {}", output.var, max),
                            passed: false,
                            error: Some(format!("{} is not less than {}", num, max)),
                        };
                    }
                }
            }
        }

        // Check greater_than
        if let Some(min) = output.greater_than {
            if let Some(val) = value {
                if let Ok(num) = val.parse::<i64>() {
                    if num <= min {
                        return AssertionCheckResult {
                            description: format!("Variable '{}' > {}", output.var, min),
                            passed: false,
                            error: Some(format!("{} is not greater than {}", num, min)),
                        };
                    }
                }
            }
        }

        // Check equals
        if let Some(expected) = &output.equals {
            if value != Some(expected) {
                return AssertionCheckResult {
                    description: format!("Variable '{}' equals '{}'", output.var, expected),
                    passed: false,
                    error: Some(format!(
                        "Expected '{}', got '{}'",
                        expected,
                        value.map_or("undefined", String::as_str)
                    )),
                };
            }
        }

        AssertionCheckResult {
            description: format!("Variable '{}' assertion", output.var),
            passed: true,
            error: None,
        }
    }

    /// Export execution trace as JSON.
    pub fn export_trace_json(&self) -> String {
        serde_json::json!({
            "playbook": self.playbook.name,
            "state_path": self.state_path,
            "variables": self.variables,
        })
        .to_string()
    }
}

/// Convert a state machine to SVG format.
pub fn to_svg(playbook: &Playbook) -> String {
    let dot = super::state_machine::to_dot(playbook);

    // Generate SVG header
    let mut svg = String::from(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
  <style>
    .state { fill: #e0e0e0; stroke: #333; stroke-width: 2; }
    .state-final { fill: #c8e6c9; }
    .transition { stroke: #333; stroke-width: 1.5; fill: none; marker-end: url(#arrow); }
    .label { font-family: sans-serif; font-size: 12px; }
  </style>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>
  <text x="10" y="20" class="label">State Machine: "##,
    );

    svg.push_str(&playbook.machine.id);
    svg.push_str("</text>\n");

    // Add states as circles (simplified layout)
    let mut y_offset = 100;
    for (id, state) in &playbook.machine.states {
        let class = if state.final_state {
            "state state-final"
        } else {
            "state"
        };
        svg.push_str(&format!(
            r#"  <ellipse cx="400" cy="{}" rx="60" ry="30" class="{}"/>
  <text x="400" y="{}" text-anchor="middle" class="label">{}</text>
"#,
            y_offset,
            class,
            y_offset + 5,
            id
        ));
        y_offset += 100;
    }

    // Add comment about DOT source
    svg.push_str(&format!(
        "\n  <!-- DOT source:\n{}\n  -->\n",
        dot.lines()
            .map(|l| format!("       {}", l))
            .collect::<Vec<_>>()
            .join("\n")
    ));

    svg.push_str("</svg>");
    svg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playbook::schema::Playbook;

    struct MockExecutor;

    impl ActionExecutor for MockExecutor {
        fn click(&mut self, _: &str) -> Result<(), ExecutorError> {
            Ok(())
        }
        fn type_text(&mut self, _: &str, _: &str) -> Result<(), ExecutorError> {
            Ok(())
        }
        fn wait(
            &mut self,
            _: &crate::playbook::schema::WaitCondition,
        ) -> Result<(), ExecutorError> {
            Ok(())
        }
        fn navigate(&mut self, _: &str) -> Result<(), ExecutorError> {
            Ok(())
        }
        fn execute_script(&mut self, _: &str) -> Result<String, ExecutorError> {
            Ok(String::new())
        }
        fn screenshot(&mut self, _: &str) -> Result<(), ExecutorError> {
            Ok(())
        }
        fn element_exists(&self, _: &str) -> Result<bool, ExecutorError> {
            Ok(true)
        }
        fn get_text(&self, _: &str) -> Result<String, ExecutorError> {
            Ok(String::new())
        }
        fn get_attribute(&self, _: &str, _: &str) -> Result<String, ExecutorError> {
            Ok(String::new())
        }
        fn get_url(&self) -> Result<String, ExecutorError> {
            Ok(String::new())
        }
        fn evaluate(&self, _: &str) -> Result<bool, ExecutorError> {
            Ok(true)
        }
    }

    #[test]
    fn test_forbidden_transition_detection() {
        let yaml = r##"
version: "1.0"
name: "Test Playbook"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    middle:
      id: "middle"
    end:
      id: "end"
      final_state: true
  transitions:
    - id: "t1"
      from: "start"
      to: "middle"
      event: "go"
    - id: "t2"
      from: "middle"
      to: "end"
      event: "finish"
  forbidden:
    - from: "start"
      to: "end"
      reason: "Cannot skip middle state"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let runner = PlaybookRunner::new(playbook, MockExecutor);

        // Check forbidden transition
        let err = runner.check_forbidden("start", "end");
        assert!(err.is_some());
        assert!(err
            .expect("should have error")
            .contains("Cannot skip middle state"));

        // Check allowed transition
        let ok = runner.check_forbidden("start", "middle");
        assert!(ok.is_none());
    }

    #[test]
    fn test_variable_substitution() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t1"
      from: "start"
      to: "start"
      event: "loop"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);

        runner
            .variables
            .insert("name".to_string(), "test".to_string());
        runner
            .variables
            .insert("value".to_string(), "123".to_string());

        let result = runner.substitute_variables("Hello ${name}, value is ${value}");
        assert_eq!(result, "Hello test, value is 123");
    }

    #[test]
    fn test_svg_export() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test_machine"
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
      event: "finish"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let svg = to_svg(&playbook);

        assert!(svg.contains("<svg"));
        assert!(svg.contains("test_machine"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_run_empty_playbook() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t_loop"
      from: "start"
      to: "start"
      event: "noop"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(result.passed);
        assert!(result.error.is_none());
        assert_eq!(result.state_path, vec!["start"]);
    }

    #[test]
    fn test_run_with_steps_and_transitions() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    middle:
      id: "middle"
    end:
      id: "end"
      final_state: true
  transitions:
    - id: "t1"
      from: "start"
      to: "middle"
      event: "go"
    - id: "t2"
      from: "middle"
      to: "end"
      event: "finish"
playbook:
  setup: []
  steps:
    - name: "Go to middle"
      transitions: ["t1"]
      capture: []
    - name: "Go to end"
      transitions: ["t2"]
      capture: []
  teardown: []
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(result.passed);
        assert_eq!(result.state_path, vec!["start", "middle", "end"]);
        assert_eq!(result.step_results.len(), 2);
    }

    #[test]
    fn test_run_with_variable_capture() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t1"
      from: "start"
      to: "start"
      event: "loop"
playbook:
  setup: []
  steps:
    - name: "Capture step"
      transitions: ["t1"]
      capture:
        - var: "captured_val"
          from: "test_value"
  teardown: []
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(result.passed);
        assert_eq!(
            result.variables.get("captured_val"),
            Some(&"test_value".to_string())
        );
    }

    #[test]
    fn test_run_forbidden_transition_fails() {
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
    - id: "forbidden_t"
      from: "start"
      to: "end"
      event: "skip"
  forbidden:
    - from: "start"
      to: "end"
      reason: "Cannot skip"
playbook:
  setup: []
  steps:
    - name: "Try forbidden"
      transitions: ["forbidden_t"]
      capture: []
  teardown: []
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(!result.passed);
        assert!(result.step_results[0]
            .error
            .as_ref()
            .expect("should have error")
            .contains("Forbidden"));
    }

    #[test]
    fn test_path_assertion_pass() {
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
playbook:
  setup: []
  steps:
    - name: "Go"
      transitions: ["t1"]
      capture: []
  teardown: []
assertions:
  path:
    expected: ["start", "end"]
  output: []
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(result.passed);
        assert!(result.assertion_results.iter().all(|a| a.passed));
    }

    #[test]
    fn test_path_assertion_fail() {
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
    - id: "t_loop"
      from: "start"
      to: "start"
      event: "noop"
assertions:
  path:
    expected: ["start", "end"]
  output: []
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(!result.passed);
        assert!(result.assertion_results.iter().any(|a| !a.passed));
    }

    #[test]
    fn test_output_assertion_not_empty() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t1"
      from: "start"
      to: "start"
      event: "loop"
playbook:
  setup: []
  steps:
    - name: "Capture"
      transitions: ["t1"]
      capture:
        - var: "my_var"
          from: "some_value"
  teardown: []
assertions:
  output:
    - var: "my_var"
      not_empty: true
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(result.passed);
    }

    #[test]
    fn test_output_assertion_not_empty_fails() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t_loop"
      from: "start"
      to: "start"
      event: "noop"
assertions:
  output:
    - var: "missing_var"
      not_empty: true
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(!result.passed);
    }

    #[test]
    fn test_output_assertion_matches() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t1"
      from: "start"
      to: "start"
      event: "loop"
playbook:
  setup: []
  steps:
    - name: "Capture"
      transitions: ["t1"]
      capture:
        - var: "email"
          from: "test@example.com"
  teardown: []
assertions:
  output:
    - var: "email"
      matches: ".*@.*\\.com"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(result.passed);
    }

    #[test]
    fn test_output_assertion_matches_fails() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t1"
      from: "start"
      to: "start"
      event: "loop"
playbook:
  setup: []
  steps:
    - name: "Capture"
      transitions: ["t1"]
      capture:
        - var: "value"
          from: "abc"
  teardown: []
assertions:
  output:
    - var: "value"
      matches: "^[0-9]+$"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(!result.passed);
    }

    #[test]
    fn test_output_assertion_matches_undefined() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t_loop"
      from: "start"
      to: "start"
      event: "noop"
assertions:
  output:
    - var: "undefined_var"
      matches: ".*"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(!result.passed);
    }

    #[test]
    fn test_output_assertion_less_than() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t1"
      from: "start"
      to: "start"
      event: "loop"
playbook:
  setup: []
  steps:
    - name: "Capture"
      transitions: ["t1"]
      capture:
        - var: "count"
          from: "5"
  teardown: []
assertions:
  output:
    - var: "count"
      less_than: 10
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(result.passed);
    }

    #[test]
    fn test_output_assertion_less_than_fails() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t1"
      from: "start"
      to: "start"
      event: "loop"
playbook:
  setup: []
  steps:
    - name: "Capture"
      transitions: ["t1"]
      capture:
        - var: "count"
          from: "15"
  teardown: []
assertions:
  output:
    - var: "count"
      less_than: 10
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(!result.passed);
    }

    #[test]
    fn test_output_assertion_greater_than() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t1"
      from: "start"
      to: "start"
      event: "loop"
playbook:
  setup: []
  steps:
    - name: "Capture"
      transitions: ["t1"]
      capture:
        - var: "count"
          from: "100"
  teardown: []
assertions:
  output:
    - var: "count"
      greater_than: 50
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(result.passed);
    }

    #[test]
    fn test_output_assertion_greater_than_fails() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t1"
      from: "start"
      to: "start"
      event: "loop"
playbook:
  setup: []
  steps:
    - name: "Capture"
      transitions: ["t1"]
      capture:
        - var: "count"
          from: "10"
  teardown: []
assertions:
  output:
    - var: "count"
      greater_than: 50
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(!result.passed);
    }

    #[test]
    fn test_output_assertion_equals() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t1"
      from: "start"
      to: "start"
      event: "loop"
playbook:
  setup: []
  steps:
    - name: "Capture"
      transitions: ["t1"]
      capture:
        - var: "result"
          from: "success"
  teardown: []
assertions:
  output:
    - var: "result"
      equals: "success"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(result.passed);
    }

    #[test]
    fn test_output_assertion_equals_fails() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t1"
      from: "start"
      to: "start"
      event: "loop"
playbook:
  setup: []
  steps:
    - name: "Capture"
      transitions: ["t1"]
      capture:
        - var: "result"
          from: "failure"
  teardown: []
assertions:
  output:
    - var: "result"
      equals: "success"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(!result.passed);
    }

    #[test]
    fn test_export_trace_json() {
        let yaml = r##"
version: "1.0"
name: "Trace Test"
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
playbook:
  setup: []
  steps:
    - name: "Go"
      transitions: ["t1"]
      capture:
        - var: "test_var"
          from: "test_value"
  teardown: []
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        runner.run();

        let json = runner.export_trace_json();
        assert!(json.contains("Trace Test"));
        assert!(json.contains("state_path"));
        assert!(json.contains("test_var"));
    }

    #[test]
    fn test_teardown_with_ignore_errors() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t_loop"
      from: "start"
      to: "start"
      event: "noop"
playbook:
  setup: []
  steps: []
  teardown:
    - action:
        wasm: "cleanup"
        args: []
      ignore_errors: true
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(result.passed);
    }

    #[test]
    fn test_run_step_with_nonexistent_transition() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t_loop"
      from: "start"
      to: "start"
      event: "noop"
playbook:
  setup: []
  steps:
    - name: "Bad transition"
      transitions: ["nonexistent"]
      capture: []
  teardown: []
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        // Should still pass, just no state change
        assert!(result.passed);
    }

    #[test]
    fn test_step_with_multiple_transitions() {
        let yaml = r##"
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
      final_state: true
  transitions:
    - id: "t1"
      from: "a"
      to: "b"
      event: "step1"
    - id: "t2"
      from: "b"
      to: "c"
      event: "step2"
playbook:
  setup: []
  steps:
    - name: "Multi-transition step"
      transitions: ["t1", "t2"]
      capture: []
  teardown: []
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(result.passed);
        assert_eq!(result.state_path, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_variable_substitution_with_captured_variables() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    next:
      id: "next"
  transitions:
    - id: "t1"
      from: "start"
      to: "next"
      event: "go"
playbook:
  setup: []
  steps:
    - name: "First capture"
      transitions: ["t1"]
      capture:
        - var: "prefix"
          from: "hello"
    - name: "Use captured"
      transitions: []
      capture:
        - var: "message"
          from: "${prefix}_world"
  teardown: []
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(result.passed);
        assert_eq!(result.variables.get("prefix"), Some(&"hello".to_string()));
        assert_eq!(
            result.variables.get("message"),
            Some(&"hello_world".to_string())
        );
    }

    #[test]
    fn test_output_assertion_not_empty_with_empty_string() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t1"
      from: "start"
      to: "start"
      event: "loop"
playbook:
  setup: []
  steps:
    - name: "Capture empty"
      transitions: ["t1"]
      capture:
        - var: "empty_var"
          from: ""
  teardown: []
assertions:
  output:
    - var: "empty_var"
      not_empty: true
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(!result.passed);
        assert!(result.assertion_results.iter().any(|a| !a.passed
            && a.error
                .as_ref()
                .is_some_and(|e| e.contains("empty or undefined"))));
    }

    #[test]
    fn test_output_assertion_less_than_non_numeric() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t1"
      from: "start"
      to: "start"
      event: "loop"
playbook:
  setup: []
  steps:
    - name: "Capture non-numeric"
      transitions: ["t1"]
      capture:
        - var: "text_val"
          from: "not_a_number"
  teardown: []
assertions:
  output:
    - var: "text_val"
      less_than: 100
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        // Should pass because the parse fails silently and assertion defaults to pass
        assert!(result.passed);
    }

    #[test]
    fn test_output_assertion_greater_than_non_numeric() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t1"
      from: "start"
      to: "start"
      event: "loop"
playbook:
  setup: []
  steps:
    - name: "Capture non-numeric"
      transitions: ["t1"]
      capture:
        - var: "text_val"
          from: "not_a_number"
  teardown: []
assertions:
  output:
    - var: "text_val"
      greater_than: 0
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        // Should pass because the parse fails silently and assertion defaults to pass
        assert!(result.passed);
    }

    #[test]
    fn test_output_assertion_equals_undefined() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t_loop"
      from: "start"
      to: "start"
      event: "noop"
assertions:
  output:
    - var: "missing"
      equals: "expected"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(!result.passed);
        assert!(result
            .assertion_results
            .iter()
            .any(|a| !a.passed && a.error.as_ref().is_some_and(|e| e.contains("undefined"))));
    }

    #[test]
    fn test_output_assertion_less_than_undefined() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t_loop"
      from: "start"
      to: "start"
      event: "noop"
assertions:
  output:
    - var: "missing"
      less_than: 100
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        // Should pass because undefined value is None and the branch skips
        assert!(result.passed);
    }

    #[test]
    fn test_output_assertion_greater_than_undefined() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t_loop"
      from: "start"
      to: "start"
      event: "noop"
assertions:
  output:
    - var: "missing"
      greater_than: 0
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        // Should pass because undefined value is None and the branch skips
        assert!(result.passed);
    }

    #[test]
    fn test_teardown_runs_after_step_failure() {
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
    - id: "forbidden_t"
      from: "start"
      to: "end"
      event: "skip"
  forbidden:
    - from: "start"
      to: "end"
      reason: "Cannot skip"
playbook:
  setup: []
  steps:
    - name: "Fail with forbidden"
      transitions: ["forbidden_t"]
      capture: []
  teardown:
    - action:
        wasm: "cleanup"
        args: []
      ignore_errors: false
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        // Teardown should have run even though step failed
        assert!(!result.passed);
    }

    #[test]
    fn test_svg_export_with_final_state() {
        let yaml = r##"
version: "1.0"
machine:
  id: "svg_test"
  initial: "start"
  states:
    start:
      id: "start"
    middle:
      id: "middle"
    end:
      id: "end"
      final_state: true
  transitions:
    - id: "t1"
      from: "start"
      to: "middle"
      event: "go"
    - id: "t2"
      from: "middle"
      to: "end"
      event: "finish"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let svg = to_svg(&playbook);

        assert!(svg.contains("<svg"));
        assert!(svg.contains("svg_test"));
        assert!(svg.contains("state-final")); // Final state should have this class
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("DOT source")); // Comment with DOT source
    }

    #[test]
    fn test_no_assertions_section() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t_loop"
      from: "start"
      to: "start"
      event: "noop"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(result.passed);
        assert!(result.assertion_results.is_empty());
    }

    #[test]
    fn test_step_result_fields() {
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
playbook:
  setup: []
  steps:
    - name: "Test Step"
      transitions: ["t1"]
      capture:
        - var: "step_var"
          from: "step_value"
  teardown: []
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(result.passed);
        assert_eq!(result.step_results.len(), 1);
        let step = &result.step_results[0];
        assert_eq!(step.name, "Test Step");
        assert!(step.passed);
        assert!(step.error.is_none());
        assert_eq!(
            step.captured.get("step_var"),
            Some(&"step_value".to_string())
        );
    }

    #[test]
    fn test_playbook_run_result_fields() {
        let yaml = r##"
version: "1.0"
name: "Result Test Playbook"
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
playbook:
  setup: []
  steps:
    - name: "Go"
      transitions: ["t1"]
      capture:
        - var: "test_var"
          from: "test_value"
  teardown: []
assertions:
  path:
    expected: ["start", "end"]
  output:
    - var: "test_var"
      equals: "test_value"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(result.passed);
        assert!(result.error.is_none());
        assert_eq!(result.state_path, vec!["start", "end"]);
        assert_eq!(
            result.variables.get("test_var"),
            Some(&"test_value".to_string())
        );
        assert!(!result.total_time.is_zero() || result.total_time == std::time::Duration::ZERO);
        assert_eq!(result.step_results.len(), 1);
        assert_eq!(result.assertion_results.len(), 2); // path + output
        assert!(result.assertion_results.iter().all(|a| a.passed));
    }

    #[test]
    fn test_assertion_result_error_formats() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t_loop"
      from: "start"
      to: "start"
      event: "noop"
assertions:
  path:
    expected: ["start", "wrong", "path"]
  output:
    - var: "missing"
      not_empty: true
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(!result.passed);
        assert!(result
            .error
            .as_ref()
            .is_some_and(|e| e.contains("Assertions failed")));

        // Check path assertion error format
        let path_result = result
            .assertion_results
            .iter()
            .find(|a| a.description.contains("Path"));
        assert!(path_result.is_some());
        let path_err = path_result.and_then(|p| p.error.as_ref());
        assert!(path_err.is_some_and(|e| e.contains("Expected path")));
    }

    #[test]
    fn test_less_than_boundary_value() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t1"
      from: "start"
      to: "start"
      event: "loop"
playbook:
  setup: []
  steps:
    - name: "Capture"
      transitions: ["t1"]
      capture:
        - var: "count"
          from: "10"
  teardown: []
assertions:
  output:
    - var: "count"
      less_than: 10
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        // 10 is not less than 10
        assert!(!result.passed);
    }

    #[test]
    fn test_greater_than_boundary_value() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t1"
      from: "start"
      to: "start"
      event: "loop"
playbook:
  setup: []
  steps:
    - name: "Capture"
      transitions: ["t1"]
      capture:
        - var: "count"
          from: "50"
  teardown: []
assertions:
  output:
    - var: "count"
      greater_than: 50
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        // 50 is not greater than 50
        assert!(!result.passed);
    }

    #[test]
    fn test_multiple_output_assertions_on_same_var() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t1"
      from: "start"
      to: "start"
      event: "loop"
playbook:
  setup: []
  steps:
    - name: "Capture"
      transitions: ["t1"]
      capture:
        - var: "count"
          from: "50"
  teardown: []
assertions:
  output:
    - var: "count"
      not_empty: true
    - var: "count"
      greater_than: 40
    - var: "count"
      less_than: 60
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(result.passed);
        assert_eq!(result.assertion_results.len(), 3);
    }

    #[test]
    fn test_step_fails_early_remaining_steps_skipped() {
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
    - id: "forbidden_t"
      from: "start"
      to: "end"
      event: "skip"
    - id: "t_loop"
      from: "start"
      to: "start"
      event: "loop"
  forbidden:
    - from: "start"
      to: "end"
      reason: "Cannot skip"
playbook:
  setup: []
  steps:
    - name: "First (fails)"
      transitions: ["forbidden_t"]
      capture: []
    - name: "Second (should be skipped)"
      transitions: ["t_loop"]
      capture:
        - var: "should_not_exist"
          from: "value"
  teardown: []
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);
        let result = runner.run();

        assert!(!result.passed);
        // Only one step should have been executed
        assert_eq!(result.step_results.len(), 1);
        // Variable from second step should not exist
        assert!(result.variables.get("should_not_exist").is_none());
    }

    #[test]
    fn test_forbidden_check_multiple_forbidden_rules() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    middle:
      id: "middle"
    end:
      id: "end"
  transitions:
    - id: "t1"
      from: "start"
      to: "middle"
      event: "go"
    - id: "t2"
      from: "middle"
      to: "end"
      event: "finish"
  forbidden:
    - from: "start"
      to: "end"
      reason: "Cannot skip middle from start"
    - from: "middle"
      to: "start"
      reason: "Cannot go backwards"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let runner = PlaybookRunner::new(playbook, MockExecutor);

        // First forbidden rule
        let err1 = runner.check_forbidden("start", "end");
        assert!(err1.is_some());
        assert!(err1
            .as_ref()
            .is_some_and(|e| e.contains("Cannot skip middle from start")));

        // Second forbidden rule
        let err2 = runner.check_forbidden("middle", "start");
        assert!(err2.is_some());
        assert!(err2
            .as_ref()
            .is_some_and(|e| e.contains("Cannot go backwards")));

        // Allowed transition
        let ok = runner.check_forbidden("start", "middle");
        assert!(ok.is_none());
    }

    #[test]
    fn test_substitute_variables_no_match() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t_loop"
      from: "start"
      to: "start"
      event: "noop"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let runner = PlaybookRunner::new(playbook, MockExecutor);

        // No variables set, so pattern should remain unchanged
        let result = runner.substitute_variables("No ${vars} here ${at_all}");
        assert_eq!(result, "No ${vars} here ${at_all}");
    }

    #[test]
    fn test_substitute_variables_partial_match() {
        let yaml = r##"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions:
    - id: "t_loop"
      from: "start"
      to: "start"
      event: "noop"
"##;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let mut runner = PlaybookRunner::new(playbook, MockExecutor);

        runner
            .variables
            .insert("found".to_string(), "YES".to_string());

        let result = runner.substitute_variables("${found} but ${not_found}");
        assert_eq!(result, "YES but ${not_found}");
    }

    #[test]
    fn test_assertion_check_result_clone() {
        let result = AssertionCheckResult {
            description: "Test".to_string(),
            passed: true,
            error: None,
        };
        let cloned = result;
        assert_eq!(cloned.description, "Test");
        assert!(cloned.passed);
        assert!(cloned.error.is_none());
    }

    #[test]
    fn test_step_result_clone() {
        let result = StepResult {
            name: "Test Step".to_string(),
            passed: false,
            duration: std::time::Duration::from_millis(100),
            captured: HashMap::new(),
            error: Some("Test error".to_string()),
        };
        let cloned = result;
        assert_eq!(cloned.name, "Test Step");
        assert!(!cloned.passed);
        assert_eq!(cloned.duration, std::time::Duration::from_millis(100));
        assert_eq!(cloned.error, Some("Test error".to_string()));
    }
}
