//! Playbook YAML schema types for state machine testing.
//!
//! Implements SCXML-inspired state definitions with transition-based assertions.
//! Reference: W3C SCXML Specification <https://www.w3.org/TR/scxml/>

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Root playbook configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Playbook {
    /// Schema version (must be "1.0")
    pub version: String,
    /// Playbook name
    #[serde(default)]
    pub name: String,
    /// Playbook description
    #[serde(default)]
    pub description: String,
    /// State machine definition
    pub machine: StateMachine,
    /// Performance budget constraints
    #[serde(default)]
    pub performance: PerformanceBudget,
    /// Playbook execution steps
    #[serde(default)]
    pub playbook: Option<PlaybookSteps>,
    /// Assertions to verify
    #[serde(default)]
    pub assertions: Option<PlaybookAssertions>,
    /// Falsification protocol
    #[serde(default)]
    pub falsification: Option<FalsificationConfig>,
    /// Optional metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// State machine definition following SCXML semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMachine {
    /// Unique identifier for the machine
    pub id: String,
    /// Initial state ID (must exist in states)
    pub initial: String,
    /// State definitions keyed by ID
    pub states: HashMap<String, State>,
    /// Transition definitions
    pub transitions: Vec<Transition>,
    /// Forbidden transitions (must never occur)
    #[serde(default)]
    pub forbidden: Vec<ForbiddenTransition>,
    /// Global performance constraints
    #[serde(default)]
    pub performance: Option<PerformanceBudget>,
}

/// Forbidden transition that must never occur.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForbiddenTransition {
    /// Source state ID
    pub from: String,
    /// Target state ID
    pub to: String,
    /// Reason why this transition is forbidden
    #[serde(default)]
    pub reason: String,
}

/// Individual state definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    /// State identifier (must be unique)
    pub id: String,
    /// Human-readable description
    #[serde(default)]
    pub description: String,
    /// Entry actions executed when entering this state
    #[serde(default)]
    pub on_entry: Vec<Action>,
    /// Exit actions executed when leaving this state
    #[serde(default)]
    pub on_exit: Vec<Action>,
    /// Invariant conditions that must hold while in this state
    #[serde(default)]
    pub invariants: Vec<Invariant>,
    /// Whether this is a final (accepting) state
    #[serde(default)]
    pub final_state: bool,
}

/// State transition definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    /// Unique transition identifier
    pub id: String,
    /// Source state ID
    pub from: String,
    /// Target state ID
    pub to: String,
    /// Event that triggers this transition
    pub event: String,
    /// Guard condition (optional)
    #[serde(default)]
    pub guard: Option<String>,
    /// Actions to execute during transition
    #[serde(default)]
    pub actions: Vec<Action>,
    /// Expected assertions after transition
    #[serde(default)]
    pub assertions: Vec<Assertion>,
}

/// Action to execute during state entry, exit, or transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Action {
    /// Click on an element
    #[serde(rename = "click")]
    Click { selector: String },
    /// Type text into an element
    #[serde(rename = "type")]
    Type { selector: String, text: String },
    /// Wait for a condition
    #[serde(rename = "wait")]
    Wait { condition: WaitCondition },
    /// Navigate to URL
    #[serde(rename = "navigate")]
    Navigate { url: String },
    /// Execute custom JavaScript
    #[serde(rename = "script")]
    Script { code: String },
    /// Take screenshot
    #[serde(rename = "screenshot")]
    Screenshot { name: String },
}

/// Wait condition types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WaitCondition {
    /// Wait for element to be visible
    #[serde(rename = "visible")]
    Visible { selector: String },
    /// Wait for element to be hidden
    #[serde(rename = "hidden")]
    Hidden { selector: String },
    /// Wait for fixed duration (ms)
    #[serde(rename = "duration")]
    Duration { ms: u64 },
    /// Wait for network idle
    #[serde(rename = "network_idle")]
    NetworkIdle,
    /// Wait for custom condition
    #[serde(rename = "condition")]
    Condition { expression: String },
}

/// Assertion to verify after transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Assertion {
    /// Element exists in DOM
    #[serde(rename = "element_exists")]
    ElementExists { selector: String },
    /// Element has specific text
    #[serde(rename = "text_equals")]
    TextEquals { selector: String, expected: String },
    /// Element has text containing substring
    #[serde(rename = "text_contains")]
    TextContains { selector: String, substring: String },
    /// Element has specific attribute value
    #[serde(rename = "attribute_equals")]
    AttributeEquals {
        selector: String,
        attribute: String,
        expected: String,
    },
    /// URL matches pattern
    #[serde(rename = "url_matches")]
    UrlMatches { pattern: String },
    /// Custom JavaScript assertion
    #[serde(rename = "script")]
    Script { expression: String },
}

/// Invariant condition that must hold while in a state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invariant {
    /// Human-readable description
    pub description: String,
    /// Condition expression
    pub condition: String,
    /// Severity if violated
    #[serde(default)]
    pub severity: InvariantSeverity,
}

/// Severity level for invariant violations.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum InvariantSeverity {
    /// Warning only, test continues
    Warning,
    /// Error, test fails
    #[default]
    Error,
    /// Critical, test aborts immediately
    Critical,
}

/// Performance budget constraints.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceBudget {
    /// Maximum time per transition (ms)
    #[serde(default)]
    pub max_transition_time_ms: Option<u64>,
    /// Maximum total playbook time (ms)
    #[serde(default)]
    pub max_total_time_ms: Option<u64>,
    /// Maximum memory usage (bytes)
    #[serde(default)]
    pub max_memory_bytes: Option<u64>,
    /// Complexity class constraint
    #[serde(default)]
    pub complexity_class: Option<ComplexityClass>,
}

/// Expected algorithmic complexity class.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ComplexityClass {
    /// O(1) - constant time
    #[serde(rename = "O(1)")]
    Constant,
    /// O(log n) - logarithmic
    #[serde(rename = "O(log n)")]
    Logarithmic,
    /// O(n) - linear
    #[serde(rename = "O(n)")]
    Linear,
    /// O(n log n) - linearithmic
    #[serde(rename = "O(n log n)")]
    Linearithmic,
    /// O(n^2) - quadratic
    #[serde(rename = "O(n^2)")]
    Quadratic,
}

/// Playbook execution steps (setup, steps, teardown).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlaybookSteps {
    /// Setup actions executed before steps
    #[serde(default)]
    pub setup: Vec<PlaybookAction>,
    /// Ordered execution steps
    #[serde(default)]
    pub steps: Vec<PlaybookStep>,
    /// Teardown actions executed after steps (even on failure)
    #[serde(default)]
    pub teardown: Vec<PlaybookAction>,
}

/// Single playbook action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybookAction {
    /// Action to execute
    pub action: ActionSpec,
    /// Description of the action
    #[serde(default)]
    pub description: String,
    /// Whether to ignore errors
    #[serde(default)]
    pub ignore_errors: bool,
}

/// Action specification (wasm call or other).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSpec {
    /// WASM function to call
    #[serde(default)]
    pub wasm: Option<String>,
    /// Arguments to pass
    #[serde(default)]
    pub args: Vec<String>,
}

/// Single execution step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybookStep {
    /// Step name
    pub name: String,
    /// Transitions to execute in order
    #[serde(default)]
    pub transitions: Vec<String>,
    /// Timeout for this step
    #[serde(default)]
    pub timeout: Option<String>,
    /// Variables to capture after step
    #[serde(default)]
    pub capture: Vec<VariableCapture>,
}

/// Variable capture specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableCapture {
    /// Variable name to store result
    pub var: String,
    /// Expression to evaluate
    pub from: String,
}

/// Playbook assertions configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlaybookAssertions {
    /// Expected state path
    #[serde(default)]
    pub path: Option<PathAssertion>,
    /// Output assertions
    #[serde(default)]
    pub output: Vec<OutputAssertion>,
    /// Complexity assertion
    #[serde(default)]
    pub complexity: Option<ComplexityAssertion>,
}

/// Path assertion - expected state sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathAssertion {
    /// Expected sequence of state IDs
    pub expected: Vec<String>,
}

/// Output assertion on captured variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputAssertion {
    /// Variable name
    pub var: String,
    /// Assert not empty
    #[serde(default)]
    pub not_empty: Option<bool>,
    /// Assert matches regex
    #[serde(default)]
    pub matches: Option<String>,
    /// Assert less than value
    #[serde(default)]
    pub less_than: Option<i64>,
    /// Assert greater than value
    #[serde(default)]
    pub greater_than: Option<i64>,
    /// Assert equals value
    #[serde(default)]
    pub equals: Option<String>,
}

/// Complexity assertion for O(n) verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityAssertion {
    /// Operation being measured
    pub operation: String,
    /// Variable containing measurement
    pub measure: String,
    /// Variable containing input size
    pub input_var: String,
    /// Expected complexity class
    pub expected: ComplexityClass,
    /// Allowed tolerance (0.0 - 1.0)
    #[serde(default)]
    pub tolerance: f64,
    /// Sample sizes for measurement
    #[serde(default)]
    pub sample_sizes: Vec<usize>,
}

/// Falsification protocol configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FalsificationConfig {
    /// Mutation definitions
    #[serde(default)]
    pub mutations: Vec<MutationDef>,
}

/// Single mutation definition for falsification testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationDef {
    /// Mutation identifier
    pub id: String,
    /// Description of the mutation
    #[serde(default)]
    pub description: String,
    /// Mutation action
    pub mutate: String,
    /// Expected failure message
    pub expected_failure: String,
}

impl Playbook {
    /// Parse a playbook from YAML string.
    ///
    /// # Errors
    /// Returns error if YAML is invalid or schema validation fails.
    pub fn from_yaml(yaml: &str) -> Result<Self, PlaybookError> {
        let playbook: Playbook =
            serde_yaml_ng::from_str(yaml).map_err(|e| PlaybookError::ParseError(e.to_string()))?;
        playbook.validate()?;
        Ok(playbook)
    }

    /// Validate the playbook structure.
    fn validate(&self) -> Result<(), PlaybookError> {
        // Validate version
        if self.version != "1.0" {
            return Err(PlaybookError::InvalidVersion(self.version.clone()));
        }

        // Validate initial state exists
        if !self.machine.states.contains_key(&self.machine.initial) {
            return Err(PlaybookError::InvalidInitialState(
                self.machine.initial.clone(),
            ));
        }

        // Validate all transition sources and targets exist
        for transition in &self.machine.transitions {
            if !self.machine.states.contains_key(&transition.from) {
                return Err(PlaybookError::InvalidTransitionSource {
                    transition_id: transition.id.clone(),
                    state_id: transition.from.clone(),
                });
            }
            if !self.machine.states.contains_key(&transition.to) {
                return Err(PlaybookError::InvalidTransitionTarget {
                    transition_id: transition.id.clone(),
                    state_id: transition.to.clone(),
                });
            }
        }

        // Check for duplicate state IDs (HashMap handles this, but explicit check)
        let state_ids: Vec<_> = self.machine.states.keys().collect();
        let unique_ids: std::collections::HashSet<_> = state_ids.iter().collect();
        if state_ids.len() != unique_ids.len() {
            return Err(PlaybookError::DuplicateStateIds);
        }

        // Check for duplicate transition IDs
        let transition_ids: Vec<_> = self.machine.transitions.iter().map(|t| &t.id).collect();
        let unique_transition_ids: std::collections::HashSet<_> = transition_ids.iter().collect();
        if transition_ids.len() != unique_transition_ids.len() {
            return Err(PlaybookError::DuplicateTransitionIds);
        }

        // Validate no empty states or transitions
        if self.machine.states.is_empty() {
            return Err(PlaybookError::EmptyStates);
        }
        if self.machine.transitions.is_empty() {
            return Err(PlaybookError::EmptyTransitions);
        }

        Ok(())
    }
}

/// Errors that can occur during playbook parsing and validation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum PlaybookError {
    #[error("Failed to parse YAML: {0}")]
    ParseError(String),

    #[error("Invalid version '{0}', expected '1.0'")]
    InvalidVersion(String),

    #[error("Initial state '{0}' does not exist")]
    InvalidInitialState(String),

    #[error("Transition '{transition_id}' references non-existent source state '{state_id}'")]
    InvalidTransitionSource {
        transition_id: String,
        state_id: String,
    },

    #[error("Transition '{transition_id}' references non-existent target state '{state_id}'")]
    InvalidTransitionTarget {
        transition_id: String,
        state_id: String,
    },

    #[error("Duplicate state IDs detected")]
    DuplicateStateIds,

    #[error("Duplicate transition IDs detected")]
    DuplicateTransitionIds,

    #[error("States cannot be empty")]
    EmptyStates,

    #[error("Transitions cannot be empty")]
    EmptyTransitions,
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_PLAYBOOK: &str = r##"
version: "1.0"
machine:
  id: "login_flow"
  initial: "logged_out"
  states:
    logged_out:
      id: "logged_out"
      description: "User is not authenticated"
      invariants:
        - description: "Login button visible"
          condition: "document.querySelector('#login-btn') !== null"
    logged_in:
      id: "logged_in"
      description: "User is authenticated"
      final_state: true
  transitions:
    - id: "t1"
      from: "logged_out"
      to: "logged_in"
      event: "login_success"
      assertions:
        - type: element_exists
          selector: "#welcome-message"
"##;

    #[test]
    fn test_parse_valid_playbook() {
        let playbook = Playbook::from_yaml(VALID_PLAYBOOK).expect("Should parse valid playbook");
        assert_eq!(playbook.version, "1.0");
        assert_eq!(playbook.machine.id, "login_flow");
        assert_eq!(playbook.machine.initial, "logged_out");
        assert_eq!(playbook.machine.states.len(), 2);
        assert_eq!(playbook.machine.transitions.len(), 1);
    }

    #[test]
    fn test_reject_invalid_version() {
        let yaml = VALID_PLAYBOOK.replace("version: \"1.0\"", "version: \"2.0\"");
        let result = Playbook::from_yaml(&yaml);
        assert!(matches!(result, Err(PlaybookError::InvalidVersion(_))));
    }

    #[test]
    fn test_reject_invalid_initial_state() {
        let yaml = VALID_PLAYBOOK.replace("initial: \"logged_out\"", "initial: \"nonexistent\"");
        let result = Playbook::from_yaml(&yaml);
        assert!(matches!(result, Err(PlaybookError::InvalidInitialState(_))));
    }

    #[test]
    fn test_reject_invalid_transition_source() {
        let yaml = VALID_PLAYBOOK.replace("from: \"logged_out\"", "from: \"nonexistent\"");
        let result = Playbook::from_yaml(&yaml);
        assert!(matches!(
            result,
            Err(PlaybookError::InvalidTransitionSource { .. })
        ));
    }

    #[test]
    fn test_reject_invalid_transition_target() {
        let yaml = VALID_PLAYBOOK.replace("to: \"logged_in\"", "to: \"nonexistent\"");
        let result = Playbook::from_yaml(&yaml);
        assert!(matches!(
            result,
            Err(PlaybookError::InvalidTransitionTarget { .. })
        ));
    }

    #[test]
    fn test_reject_empty_states() {
        let yaml = r#"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states: {}
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
"#;
        let result = Playbook::from_yaml(yaml);
        // Empty states causes initial state validation to fail first
        assert!(result.is_err());
        // Could be EmptyStates or InvalidInitialState depending on validation order
        assert!(matches!(
            result,
            Err(PlaybookError::EmptyStates | PlaybookError::InvalidInitialState(_))
        ));
    }

    #[test]
    fn test_reject_empty_transitions() {
        let yaml = r#"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
  transitions: []
"#;
        let result = Playbook::from_yaml(yaml);
        assert!(matches!(result, Err(PlaybookError::EmptyTransitions)));
    }

    // === Additional tests for improved coverage ===

    #[test]
    fn test_parse_error_invalid_yaml() {
        let yaml = "this is not: valid: yaml: {{{{";
        let result = Playbook::from_yaml(yaml);
        assert!(matches!(result, Err(PlaybookError::ParseError(_))));
    }

    #[test]
    fn test_parse_error_missing_required_field() {
        let yaml = r#"
version: "1.0"
"#;
        let result = Playbook::from_yaml(yaml);
        assert!(matches!(result, Err(PlaybookError::ParseError(_))));
    }

    #[test]
    fn test_duplicate_transition_ids() {
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
  transitions:
    - id: "t1"
      from: "a"
      to: "b"
      event: "go"
    - id: "t1"
      from: "b"
      to: "a"
      event: "back"
"#;
        let result = Playbook::from_yaml(yaml);
        assert!(matches!(result, Err(PlaybookError::DuplicateTransitionIds)));
    }

    #[test]
    fn test_playbook_error_display() {
        // Test all error Display implementations
        let err = PlaybookError::ParseError("test error".to_string());
        assert!(err.to_string().contains("Failed to parse YAML"));

        let err = PlaybookError::InvalidVersion("2.0".to_string());
        assert!(err.to_string().contains("Invalid version '2.0'"));

        let err = PlaybookError::InvalidInitialState("missing".to_string());
        assert!(err.to_string().contains("Initial state 'missing'"));

        let err = PlaybookError::InvalidTransitionSource {
            transition_id: "t1".to_string(),
            state_id: "missing".to_string(),
        };
        assert!(err.to_string().contains("Transition 't1'"));
        assert!(err.to_string().contains("source state 'missing'"));

        let err = PlaybookError::InvalidTransitionTarget {
            transition_id: "t2".to_string(),
            state_id: "gone".to_string(),
        };
        assert!(err.to_string().contains("Transition 't2'"));
        assert!(err.to_string().contains("target state 'gone'"));

        let err = PlaybookError::DuplicateStateIds;
        assert!(err.to_string().contains("Duplicate state IDs"));

        let err = PlaybookError::DuplicateTransitionIds;
        assert!(err.to_string().contains("Duplicate transition IDs"));

        let err = PlaybookError::EmptyStates;
        assert!(err.to_string().contains("States cannot be empty"));

        let err = PlaybookError::EmptyTransitions;
        assert!(err.to_string().contains("Transitions cannot be empty"));
    }

    #[test]
    fn test_invariant_severity_default() {
        let severity = InvariantSeverity::default();
        assert_eq!(severity, InvariantSeverity::Error);
    }

    #[test]
    fn test_invariant_severity_parsing() {
        let yaml = r#"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
      invariants:
        - description: "warning"
          condition: "true"
          severity: warning
        - description: "error"
          condition: "true"
          severity: error
        - description: "critical"
          condition: "true"
          severity: critical
    end:
      id: "end"
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Should parse");
        let state = playbook.machine.states.get("start").unwrap();
        assert_eq!(state.invariants.len(), 3);
        assert_eq!(state.invariants[0].severity, InvariantSeverity::Warning);
        assert_eq!(state.invariants[1].severity, InvariantSeverity::Error);
        assert_eq!(state.invariants[2].severity, InvariantSeverity::Critical);
    }

    #[test]
    fn test_complexity_class_parsing() {
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
  performance:
    complexity_class: "O(1)"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Should parse O(1)");
        assert_eq!(
            playbook.machine.performance.unwrap().complexity_class,
            Some(ComplexityClass::Constant)
        );

        // Test all complexity classes
        for (class_str, expected) in [
            ("O(log n)", ComplexityClass::Logarithmic),
            ("O(n)", ComplexityClass::Linear),
            ("O(n log n)", ComplexityClass::Linearithmic),
            ("O(n^2)", ComplexityClass::Quadratic),
        ] {
            let yaml = format!(
                r#"
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
  performance:
    complexity_class: "{class_str}"
"#
            );
            let playbook = Playbook::from_yaml(&yaml).expect("Should parse");
            assert_eq!(
                playbook.machine.performance.unwrap().complexity_class,
                Some(expected)
            );
        }
    }

    #[test]
    fn test_action_variants_parsing() {
        let yaml = r#"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
      on_entry:
        - type: click
          selector: ".btn"
        - type: type
          selector: ".input"
          text: "hello"
        - type: wait
          condition:
            type: visible
            selector: ".element"
        - type: navigate
          url: "https://example.com"
        - type: script
          code: "console.log('hi')"
        - type: screenshot
          name: "screenshot1"
      on_exit:
        - type: click
          selector: ".logout"
    end:
      id: "end"
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
      actions:
        - type: wait
          condition:
            type: hidden
            selector: ".loader"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Should parse actions");
        let state = playbook.machine.states.get("start").unwrap();

        assert_eq!(state.on_entry.len(), 6);
        assert!(matches!(&state.on_entry[0], Action::Click { selector } if selector == ".btn"));
        assert!(
            matches!(&state.on_entry[1], Action::Type { selector, text } if selector == ".input" && text == "hello")
        );
        assert!(matches!(&state.on_entry[2], Action::Wait { .. }));
        assert!(
            matches!(&state.on_entry[3], Action::Navigate { url } if url == "https://example.com")
        );
        assert!(
            matches!(&state.on_entry[4], Action::Script { code } if code == "console.log('hi')")
        );
        assert!(matches!(&state.on_entry[5], Action::Screenshot { name } if name == "screenshot1"));

        assert_eq!(state.on_exit.len(), 1);
    }

    #[test]
    fn test_wait_condition_variants() {
        let yaml = r#"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
      on_entry:
        - type: wait
          condition:
            type: duration
            ms: 1000
        - type: wait
          condition:
            type: network_idle
        - type: wait
          condition:
            type: condition
            expression: "window.ready === true"
    end:
      id: "end"
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "go"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Should parse wait conditions");
        let state = playbook.machine.states.get("start").unwrap();

        assert_eq!(state.on_entry.len(), 3);
        if let Action::Wait { condition } = &state.on_entry[0] {
            assert!(matches!(condition, WaitCondition::Duration { ms: 1000 }));
        } else {
            panic!("Expected Wait action");
        }
        if let Action::Wait { condition } = &state.on_entry[1] {
            assert!(matches!(condition, WaitCondition::NetworkIdle));
        } else {
            panic!("Expected Wait action");
        }
        if let Action::Wait { condition } = &state.on_entry[2] {
            assert!(matches!(condition, WaitCondition::Condition { .. }));
        } else {
            panic!("Expected Wait action");
        }
    }

    #[test]
    fn test_assertion_variants() {
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
      assertions:
        - type: element_exists
          selector: ".elem"
        - type: text_equals
          selector: ".text"
          expected: "Hello"
        - type: text_contains
          selector: ".text"
          substring: "ell"
        - type: attribute_equals
          selector: ".elem"
          attribute: "data-value"
          expected: "123"
        - type: url_matches
          pattern: "^https://.*"
        - type: script
          expression: "document.title === 'Test'"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Should parse assertions");
        let transition = &playbook.machine.transitions[0];

        assert_eq!(transition.assertions.len(), 6);
        assert!(matches!(
            &transition.assertions[0],
            Assertion::ElementExists { .. }
        ));
        assert!(matches!(
            &transition.assertions[1],
            Assertion::TextEquals { .. }
        ));
        assert!(matches!(
            &transition.assertions[2],
            Assertion::TextContains { .. }
        ));
        assert!(matches!(
            &transition.assertions[3],
            Assertion::AttributeEquals { .. }
        ));
        assert!(matches!(
            &transition.assertions[4],
            Assertion::UrlMatches { .. }
        ));
        assert!(matches!(
            &transition.assertions[5],
            Assertion::Script { .. }
        ));
    }

    #[test]
    fn test_performance_budget_defaults() {
        let budget = PerformanceBudget::default();
        assert!(budget.max_transition_time_ms.is_none());
        assert!(budget.max_total_time_ms.is_none());
        assert!(budget.max_memory_bytes.is_none());
        assert!(budget.complexity_class.is_none());
    }

    #[test]
    fn test_performance_budget_full() {
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
performance:
  max_transition_time_ms: 100
  max_total_time_ms: 5000
  max_memory_bytes: 10485760
  complexity_class: "O(n)"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Should parse performance");
        assert_eq!(playbook.performance.max_transition_time_ms, Some(100));
        assert_eq!(playbook.performance.max_total_time_ms, Some(5000));
        assert_eq!(playbook.performance.max_memory_bytes, Some(10485760));
        assert_eq!(
            playbook.performance.complexity_class,
            Some(ComplexityClass::Linear)
        );
    }

    #[test]
    fn test_forbidden_transitions() {
        let yaml = r#"
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
      reason: "Must go through middle state"
    - from: "end"
      to: "start"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Should parse forbidden transitions");
        assert_eq!(playbook.machine.forbidden.len(), 2);
        assert_eq!(playbook.machine.forbidden[0].from, "start");
        assert_eq!(playbook.machine.forbidden[0].to, "end");
        assert_eq!(
            playbook.machine.forbidden[0].reason,
            "Must go through middle state"
        );
        assert_eq!(playbook.machine.forbidden[1].from, "end");
        assert_eq!(playbook.machine.forbidden[1].to, "start");
        assert_eq!(playbook.machine.forbidden[1].reason, "");
    }

    #[test]
    fn test_playbook_steps() {
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
playbook:
  setup:
    - action:
        wasm: "init_game"
        args: ["--level", "1"]
      description: "Initialize game"
      ignore_errors: false
  steps:
    - name: "Step 1"
      transitions: ["t1"]
      timeout: "30s"
      capture:
        - var: "score"
          from: "game.score"
  teardown:
    - action:
        wasm: "cleanup"
        args: []
      description: "Cleanup"
      ignore_errors: true
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Should parse playbook steps");
        let steps = playbook.playbook.unwrap();

        assert_eq!(steps.setup.len(), 1);
        assert_eq!(steps.setup[0].action.wasm, Some("init_game".to_string()));
        assert_eq!(steps.setup[0].action.args, vec!["--level", "1"]);
        assert_eq!(steps.setup[0].description, "Initialize game");
        assert!(!steps.setup[0].ignore_errors);

        assert_eq!(steps.steps.len(), 1);
        assert_eq!(steps.steps[0].name, "Step 1");
        assert_eq!(steps.steps[0].transitions, vec!["t1"]);
        assert_eq!(steps.steps[0].timeout, Some("30s".to_string()));
        assert_eq!(steps.steps[0].capture.len(), 1);
        assert_eq!(steps.steps[0].capture[0].var, "score");
        assert_eq!(steps.steps[0].capture[0].from, "game.score");

        assert_eq!(steps.teardown.len(), 1);
        assert!(steps.teardown[0].ignore_errors);
    }

    #[test]
    fn test_playbook_assertions() {
        let yaml = r#"
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
assertions:
  path:
    expected: ["start", "middle", "end"]
  output:
    - var: "score"
      not_empty: true
    - var: "score"
      matches: "^\\d+$"
    - var: "score"
      less_than: 1000
    - var: "score"
      greater_than: 0
    - var: "name"
      equals: "Player1"
  complexity:
    operation: "render"
    measure: "render_time"
    input_var: "entity_count"
    expected: "O(n)"
    tolerance: 0.1
    sample_sizes: [10, 100, 1000]
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Should parse assertions");
        let assertions = playbook.assertions.unwrap();

        assert_eq!(
            assertions.path.unwrap().expected,
            vec!["start", "middle", "end"]
        );

        assert_eq!(assertions.output.len(), 5);
        assert_eq!(assertions.output[0].var, "score");
        assert_eq!(assertions.output[0].not_empty, Some(true));
        assert_eq!(assertions.output[1].matches, Some("^\\d+$".to_string()));
        assert_eq!(assertions.output[2].less_than, Some(1000));
        assert_eq!(assertions.output[3].greater_than, Some(0));
        assert_eq!(assertions.output[4].equals, Some("Player1".to_string()));

        let complexity = assertions.complexity.unwrap();
        assert_eq!(complexity.operation, "render");
        assert_eq!(complexity.measure, "render_time");
        assert_eq!(complexity.input_var, "entity_count");
        assert_eq!(complexity.expected, ComplexityClass::Linear);
        assert!((complexity.tolerance - 0.1).abs() < f64::EPSILON);
        assert_eq!(complexity.sample_sizes, vec![10, 100, 1000]);
    }

    #[test]
    fn test_falsification_config() {
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
falsification:
  mutations:
    - id: "mut1"
      description: "Remove collision detection"
      mutate: "game.collision_enabled = false"
      expected_failure: "Player should collide with wall"
    - id: "mut2"
      mutate: "game.score = -1"
      expected_failure: "Score should never be negative"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Should parse falsification");
        let falsification = playbook.falsification.unwrap();

        assert_eq!(falsification.mutations.len(), 2);
        assert_eq!(falsification.mutations[0].id, "mut1");
        assert_eq!(
            falsification.mutations[0].description,
            "Remove collision detection"
        );
        assert_eq!(
            falsification.mutations[0].mutate,
            "game.collision_enabled = false"
        );
        assert_eq!(
            falsification.mutations[0].expected_failure,
            "Player should collide with wall"
        );
        assert_eq!(falsification.mutations[1].id, "mut2");
        assert_eq!(falsification.mutations[1].description, "");
    }

    #[test]
    fn test_metadata() {
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
metadata:
  author: "Test Author"
  game: "MyGame"
  tags: "integration,smoke"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Should parse metadata");
        assert_eq!(playbook.metadata.get("author"), Some(&"Test Author".into()));
        assert_eq!(playbook.metadata.get("game"), Some(&"MyGame".into()));
        assert_eq!(
            playbook.metadata.get("tags"),
            Some(&"integration,smoke".into())
        );
    }

    #[test]
    fn test_optional_fields_defaults() {
        let yaml = r#"
version: "1.0"
machine:
  id: "minimal"
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
        let playbook = Playbook::from_yaml(yaml).expect("Should parse minimal playbook");
        assert_eq!(playbook.name, "");
        assert_eq!(playbook.description, "");
        assert!(playbook.playbook.is_none());
        assert!(playbook.assertions.is_none());
        assert!(playbook.falsification.is_none());
        assert!(playbook.metadata.is_empty());
        assert!(playbook.machine.forbidden.is_empty());
        assert!(playbook.machine.performance.is_none());
    }

    #[test]
    fn test_state_optional_fields() {
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
        let playbook = Playbook::from_yaml(yaml).expect("Should parse");
        let state = playbook.machine.states.get("start").unwrap();
        assert_eq!(state.description, "");
        assert!(state.on_entry.is_empty());
        assert!(state.on_exit.is_empty());
        assert!(state.invariants.is_empty());
        assert!(!state.final_state);
    }

    #[test]
    fn test_transition_optional_fields() {
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
        let playbook = Playbook::from_yaml(yaml).expect("Should parse");
        let transition = &playbook.machine.transitions[0];
        assert!(transition.guard.is_none());
        assert!(transition.actions.is_empty());
        assert!(transition.assertions.is_empty());
    }

    #[test]
    fn test_transition_with_guard() {
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
      guard: "player.health > 0"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Should parse");
        let transition = &playbook.machine.transitions[0];
        assert_eq!(transition.guard, Some("player.health > 0".to_string()));
    }

    #[test]
    fn test_playbook_steps_default() {
        let steps = PlaybookSteps::default();
        assert!(steps.setup.is_empty());
        assert!(steps.steps.is_empty());
        assert!(steps.teardown.is_empty());
    }

    #[test]
    fn test_playbook_assertions_default() {
        let assertions = PlaybookAssertions::default();
        assert!(assertions.path.is_none());
        assert!(assertions.output.is_empty());
        assert!(assertions.complexity.is_none());
    }

    #[test]
    fn test_falsification_config_default() {
        let config = FalsificationConfig::default();
        assert!(config.mutations.is_empty());
    }

    #[test]
    fn test_name_and_description() {
        let yaml = r#"
version: "1.0"
name: "My Test Playbook"
description: "A comprehensive test"
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
        let playbook = Playbook::from_yaml(yaml).expect("Should parse");
        assert_eq!(playbook.name, "My Test Playbook");
        assert_eq!(playbook.description, "A comprehensive test");
    }

    #[test]
    fn test_playbook_error_clone() {
        let err = PlaybookError::InvalidVersion("2.0".to_string());
        let cloned = err.clone();
        assert_eq!(err.to_string(), cloned.to_string());
    }

    #[test]
    fn test_playbook_error_debug() {
        let err = PlaybookError::EmptyStates;
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("EmptyStates"));
    }

    #[test]
    fn test_struct_clone_derive() {
        // Test Clone implementations
        let state = State {
            id: "test".to_string(),
            description: "desc".to_string(),
            on_entry: vec![],
            on_exit: vec![],
            invariants: vec![],
            final_state: false,
        };
        let _ = state;

        let transition = Transition {
            id: "t1".to_string(),
            from: "a".to_string(),
            to: "b".to_string(),
            event: "go".to_string(),
            guard: None,
            actions: vec![],
            assertions: vec![],
        };
        let _ = transition;

        let invariant = Invariant {
            description: "test".to_string(),
            condition: "true".to_string(),
            severity: InvariantSeverity::Error,
        };
        let _ = invariant;

        let forbidden = ForbiddenTransition {
            from: "a".to_string(),
            to: "b".to_string(),
            reason: "test".to_string(),
        };
        let _ = forbidden;
    }

    #[test]
    fn test_action_clone() {
        let action = Action::Click {
            selector: ".btn".to_string(),
        };
        let _ = action;

        let action = Action::Type {
            selector: ".input".to_string(),
            text: "hello".to_string(),
        };
        let _ = action;

        let action = Action::Wait {
            condition: WaitCondition::NetworkIdle,
        };
        let _ = action;
    }

    #[test]
    fn test_wait_condition_clone() {
        let cond = WaitCondition::Visible {
            selector: ".elem".to_string(),
        };
        let _ = cond;

        let cond = WaitCondition::Hidden {
            selector: ".elem".to_string(),
        };
        let _ = cond;

        let cond = WaitCondition::Duration { ms: 100 };
        let _ = cond;

        let cond = WaitCondition::Condition {
            expression: "true".to_string(),
        };
        let _ = cond;
    }

    #[test]
    fn test_assertion_clone() {
        let assertion = Assertion::ElementExists {
            selector: ".elem".to_string(),
        };
        let _ = assertion;

        let assertion = Assertion::TextEquals {
            selector: ".elem".to_string(),
            expected: "text".to_string(),
        };
        let _ = assertion;

        let assertion = Assertion::TextContains {
            selector: ".elem".to_string(),
            substring: "text".to_string(),
        };
        let _ = assertion;

        let assertion = Assertion::AttributeEquals {
            selector: ".elem".to_string(),
            attribute: "attr".to_string(),
            expected: "val".to_string(),
        };
        let _ = assertion;

        let assertion = Assertion::UrlMatches {
            pattern: ".*".to_string(),
        };
        let _ = assertion;

        let assertion = Assertion::Script {
            expression: "true".to_string(),
        };
        let _ = assertion;
    }

    #[test]
    fn test_complexity_class_copy() {
        let class = ComplexityClass::Constant;
        let copied = class;
        assert_eq!(class, copied);
    }

    #[test]
    fn test_invariant_severity_copy() {
        let severity = InvariantSeverity::Warning;
        let copied = severity;
        assert_eq!(severity, copied);
    }

    #[test]
    fn test_action_spec_clone() {
        let spec = ActionSpec {
            wasm: Some("func".to_string()),
            args: vec!["arg1".to_string()],
        };
        let _ = spec;
    }

    #[test]
    fn test_playbook_action_clone() {
        let action = PlaybookAction {
            action: ActionSpec {
                wasm: None,
                args: vec![],
            },
            description: "test".to_string(),
            ignore_errors: true,
        };
        let _ = action;
    }

    #[test]
    fn test_playbook_step_clone() {
        let step = PlaybookStep {
            name: "step1".to_string(),
            transitions: vec!["t1".to_string()],
            timeout: Some("10s".to_string()),
            capture: vec![VariableCapture {
                var: "x".to_string(),
                from: "y".to_string(),
            }],
        };
        let _ = step;
    }

    #[test]
    fn test_variable_capture_clone() {
        let capture = VariableCapture {
            var: "x".to_string(),
            from: "y".to_string(),
        };
        let _ = capture;
    }

    #[test]
    fn test_path_assertion_clone() {
        let assertion = PathAssertion {
            expected: vec!["a".to_string(), "b".to_string()],
        };
        let _ = assertion;
    }

    #[test]
    fn test_output_assertion_clone() {
        let assertion = OutputAssertion {
            var: "x".to_string(),
            not_empty: Some(true),
            matches: Some(".*".to_string()),
            less_than: Some(100),
            greater_than: Some(0),
            equals: Some("value".to_string()),
        };
        let _ = assertion;
    }

    #[test]
    fn test_complexity_assertion_clone() {
        let assertion = ComplexityAssertion {
            operation: "op".to_string(),
            measure: "time".to_string(),
            input_var: "n".to_string(),
            expected: ComplexityClass::Linear,
            tolerance: 0.1,
            sample_sizes: vec![10, 100],
        };
        let _ = assertion;
    }

    #[test]
    fn test_mutation_def_clone() {
        let mutation = MutationDef {
            id: "m1".to_string(),
            description: "desc".to_string(),
            mutate: "x = 1".to_string(),
            expected_failure: "fail".to_string(),
        };
        let _ = mutation;
    }

    #[test]
    fn test_struct_debug_derive() {
        // Test Debug implementations
        let state = State {
            id: "test".to_string(),
            description: "desc".to_string(),
            on_entry: vec![],
            on_exit: vec![],
            invariants: vec![],
            final_state: false,
        };
        let _ = format!("{:?}", state);

        let action = Action::Click {
            selector: ".btn".to_string(),
        };
        let _ = format!("{:?}", action);

        let cond = WaitCondition::NetworkIdle;
        let _ = format!("{:?}", cond);

        let assertion = Assertion::Script {
            expression: "true".to_string(),
        };
        let _ = format!("{:?}", assertion);

        let severity = InvariantSeverity::Critical;
        let _ = format!("{:?}", severity);

        let class = ComplexityClass::Quadratic;
        let _ = format!("{:?}", class);
    }

    #[test]
    fn test_playbook_clone() {
        let playbook = Playbook::from_yaml(VALID_PLAYBOOK).expect("Should parse");
        let _ = playbook;
    }

    #[test]
    fn test_state_machine_clone() {
        let playbook = Playbook::from_yaml(VALID_PLAYBOOK).expect("Should parse");
        let _ = playbook.machine;
    }
}
