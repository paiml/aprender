//! State machine validation and analysis.
//!
//! Implements reachability analysis, orphan detection, and determinism checks
//! following TLA+ and model checking principles.
//! Reference: Lamport, "Specifying Systems" (2002)

use super::schema::{Playbook, Transition};
use std::collections::{HashMap, HashSet, VecDeque};

/// Result of state machine validation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the state machine is valid
    pub is_valid: bool,
    /// List of detected issues
    pub issues: Vec<ValidationIssue>,
    /// Reachability information
    pub reachability: ReachabilityInfo,
    /// Determinism analysis
    pub determinism: DeterminismInfo,
}

/// Information about state reachability.
#[derive(Debug, Clone, Default)]
pub struct ReachabilityInfo {
    /// States reachable from initial state
    pub reachable_states: HashSet<String>,
    /// States that cannot be reached (orphans)
    pub orphaned_states: HashSet<String>,
    /// Final states that are reachable
    pub reachable_final_states: HashSet<String>,
    /// Whether at least one final state is reachable
    pub can_reach_final: bool,
}

/// Information about state machine determinism.
#[derive(Debug, Clone, Default)]
pub struct DeterminismInfo {
    /// Whether the machine is deterministic
    pub is_deterministic: bool,
    /// Non-deterministic transitions (same source + event)
    pub non_deterministic_pairs: Vec<(String, String)>,
}

/// Types of validation issues.
#[derive(Debug, Clone)]
pub enum ValidationIssue {
    /// State is not reachable from initial state
    OrphanedState { state_id: String },
    /// No path to any final state
    NoPathToFinal { from_state: String },
    /// Dead-end state (non-final with no outgoing transitions)
    DeadEndState { state_id: String },
    /// Non-deterministic transitions
    NonDeterministic {
        state_id: String,
        event: String,
        transitions: Vec<String>,
    },
    /// Missing event handler
    UnhandledEvent { state_id: String, event: String },
    /// Self-loop without guard (potential infinite loop)
    UnguardedSelfLoop { transition_id: String },
}

impl ValidationIssue {
    /// Get the severity of this issue.
    pub fn severity(&self) -> IssueSeverity {
        match self {
            ValidationIssue::OrphanedState { .. } => IssueSeverity::Error,
            ValidationIssue::NoPathToFinal { .. } => IssueSeverity::Warning,
            ValidationIssue::DeadEndState { .. } => IssueSeverity::Error,
            ValidationIssue::NonDeterministic { .. } => IssueSeverity::Warning,
            ValidationIssue::UnhandledEvent { .. } => IssueSeverity::Info,
            ValidationIssue::UnguardedSelfLoop { .. } => IssueSeverity::Warning,
        }
    }
}

/// Severity levels for validation issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
}

/// State machine validator.
pub struct StateMachineValidator<'a> {
    playbook: &'a Playbook,
}

impl<'a> StateMachineValidator<'a> {
    /// Create a new validator for the given playbook.
    pub fn new(playbook: &'a Playbook) -> Self {
        Self { playbook }
    }

    /// Perform full validation of the state machine.
    pub fn validate(&self) -> ValidationResult {
        contract_pre_playbook_state_machine!();
        let mut issues = Vec::new();

        // Compute reachability
        let reachability = self.compute_reachability();

        // Report orphaned states
        for state_id in &reachability.orphaned_states {
            issues.push(ValidationIssue::OrphanedState {
                state_id: state_id.clone(),
            });
        }

        // Check for dead-end states
        self.check_dead_ends(&reachability, &mut issues);

        // Check determinism
        let determinism = self.check_determinism(&mut issues);

        // Check for unguarded self-loops
        self.check_self_loops(&mut issues);

        // Check paths to final states
        self.check_final_reachability(&reachability, &mut issues);

        let has_errors = issues.iter().any(|i| i.severity() == IssueSeverity::Error);

        ValidationResult {
            is_valid: !has_errors,
            issues,
            reachability,
            determinism,
        }
    }

    /// Compute which states are reachable from the initial state using BFS.
    fn compute_reachability(&self) -> ReachabilityInfo {
        contract_pre_playbook_state_machine!();
        let mut reachable = HashSet::new();
        let mut queue = VecDeque::new();

        // Start from initial state
        queue.push_back(self.playbook.machine.initial.clone());
        reachable.insert(self.playbook.machine.initial.clone());

        // BFS traversal
        while let Some(current) = queue.pop_front() {
            for transition in &self.playbook.machine.transitions {
                if transition.from == current && !reachable.contains(&transition.to) {
                    reachable.insert(transition.to.clone());
                    queue.push_back(transition.to.clone());
                }
            }
        }

        // Find orphaned states
        let all_states: HashSet<_> = self.playbook.machine.states.keys().cloned().collect();
        let orphaned: HashSet<_> = all_states.difference(&reachable).cloned().collect();

        // Find reachable final states
        let final_states: HashSet<_> = reachable
            .iter()
            .filter(|id| {
                self.playbook
                    .machine
                    .states
                    .get(*id)
                    .is_some_and(|s| s.final_state)
            })
            .cloned()
            .collect();

        ReachabilityInfo {
            reachable_states: reachable,
            orphaned_states: orphaned,
            can_reach_final: !final_states.is_empty(),
            reachable_final_states: final_states,
        }
    }

    /// Check for dead-end states (non-final states with no outgoing transitions).
    fn check_dead_ends(&self, reachability: &ReachabilityInfo, issues: &mut Vec<ValidationIssue>) {
        // Build outgoing transition map
        let mut outgoing: HashMap<&str, Vec<&Transition>> = HashMap::new();
        for transition in &self.playbook.machine.transitions {
            outgoing
                .entry(transition.from.as_str())
                .or_default()
                .push(transition);
        }

        // Check each reachable non-final state
        for state_id in &reachability.reachable_states {
            if let Some(state) = self.playbook.machine.states.get(state_id) {
                if !state.final_state && !outgoing.contains_key(state_id.as_str()) {
                    issues.push(ValidationIssue::DeadEndState {
                        state_id: state_id.clone(),
                    });
                }
            }
        }
    }

    /// Check for non-deterministic transitions.
    fn check_determinism(&self, issues: &mut Vec<ValidationIssue>) -> DeterminismInfo {
        contract_pre_playbook_state_machine!();
        // Group transitions by (source_state, event)
        let mut transition_map: HashMap<(String, String), Vec<&Transition>> = HashMap::new();

        for transition in &self.playbook.machine.transitions {
            transition_map
                .entry((transition.from.clone(), transition.event.clone()))
                .or_default()
                .push(transition);
        }

        let mut non_deterministic_pairs: Vec<(String, String)> = Vec::new();

        // Find non-deterministic cases (multiple transitions for same state+event without guards)
        for (key, trans_vec) in &transition_map {
            let (state_id, event) = key;
            if trans_vec.len() > 1 {
                // Check if all have guards (makes it deterministic)
                let all_guarded = trans_vec.iter().all(|t| t.guard.is_some());
                if !all_guarded {
                    non_deterministic_pairs.push((state_id.clone(), event.clone()));
                    issues.push(ValidationIssue::NonDeterministic {
                        state_id: state_id.clone(),
                        event: event.clone(),
                        transitions: trans_vec.iter().map(|t| t.id.clone()).collect(),
                    });
                }
            }
        }

        DeterminismInfo {
            is_deterministic: non_deterministic_pairs.is_empty(),
            non_deterministic_pairs,
        }
    }

    /// Check for unguarded self-loops.
    fn check_self_loops(&self, issues: &mut Vec<ValidationIssue>) {
        for transition in &self.playbook.machine.transitions {
            if transition.from == transition.to && transition.guard.is_none() {
                issues.push(ValidationIssue::UnguardedSelfLoop {
                    transition_id: transition.id.clone(),
                });
            }
        }
    }

    /// Check if all reachable states can reach a final state.
    fn check_final_reachability(
        &self,
        reachability: &ReachabilityInfo,
        issues: &mut Vec<ValidationIssue>,
    ) {
        if reachability.reachable_final_states.is_empty() {
            return; // No final states defined, skip this check
        }

        // Compute reverse reachability from final states
        let mut can_reach_final = reachability.reachable_final_states.clone();
        let mut changed = true;

        while changed {
            changed = false;
            for transition in &self.playbook.machine.transitions {
                if can_reach_final.contains(&transition.to)
                    && !can_reach_final.contains(&transition.from)
                {
                    can_reach_final.insert(transition.from.clone());
                    changed = true;
                }
            }
        }

        // Report states that cannot reach final
        for state_id in &reachability.reachable_states {
            if !can_reach_final.contains(state_id) {
                issues.push(ValidationIssue::NoPathToFinal {
                    from_state: state_id.clone(),
                });
            }
        }
    }
}

/// Generate a state diagram in DOT format for visualization.
pub fn to_dot(playbook: &Playbook) -> String {
    let mut dot = String::new();
    dot.push_str("digraph StateMachine {\n");
    dot.push_str("  rankdir=LR;\n");
    dot.push_str("  node [shape=ellipse];\n");

    // Mark initial state
    dot.push_str(&format!(
        "  __start [shape=point];\n  __start -> \"{}\";\n",
        playbook.machine.initial
    ));

    // Add states
    for (id, state) in &playbook.machine.states {
        let shape = if state.final_state {
            "doublecircle"
        } else {
            "ellipse"
        };
        dot.push_str(&format!("  \"{}\" [shape={}];\n", id, shape));
    }

    // Add transitions
    for transition in &playbook.machine.transitions {
        let label = if let Some(guard) = &transition.guard {
            format!("{} [{}]", transition.event, guard)
        } else {
            transition.event.clone()
        };
        dot.push_str(&format!(
            "  \"{}\" -> \"{}\" [label=\"{}\"];\n",
            transition.from, transition.to, label
        ));
    }

    dot.push_str("}\n");
    dot
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playbook::schema::Playbook;

    const VALID_PLAYBOOK: &str = r#"
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
      event: "next"
    - id: "t2"
      from: "middle"
      to: "end"
      event: "finish"
"#;

    #[test]
    fn test_valid_state_machine() {
        let playbook = Playbook::from_yaml(VALID_PLAYBOOK).expect("parse");
        let validator = StateMachineValidator::new(&playbook);
        let result = validator.validate();

        assert!(result.is_valid);
        assert!(result.reachability.orphaned_states.is_empty());
        assert!(result.reachability.can_reach_final);
        assert!(result.determinism.is_deterministic);
    }

    #[test]
    fn test_detect_orphaned_state() {
        let yaml = r#"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    orphan:
      id: "orphan"
    end:
      id: "end"
      final_state: true
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "finish"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let validator = StateMachineValidator::new(&playbook);
        let result = validator.validate();

        assert!(!result.is_valid);
        assert!(result.reachability.orphaned_states.contains("orphan"));
        assert!(result.issues.iter().any(
            |i| matches!(i, ValidationIssue::OrphanedState { state_id } if state_id == "orphan")
        ));
    }

    #[test]
    fn test_detect_dead_end() {
        let yaml = r#"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    dead_end:
      id: "dead_end"
  transitions:
    - id: "t1"
      from: "start"
      to: "dead_end"
      event: "go"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let validator = StateMachineValidator::new(&playbook);
        let result = validator.validate();

        assert!(!result.is_valid);
        assert!(result.issues.iter().any(
            |i| matches!(i, ValidationIssue::DeadEndState { state_id } if state_id == "dead_end")
        ));
    }

    #[test]
    fn test_detect_non_deterministic() {
        let yaml = r#"
version: "1.0"
machine:
  id: "test"
  initial: "start"
  states:
    start:
      id: "start"
    end1:
      id: "end1"
      final_state: true
    end2:
      id: "end2"
      final_state: true
  transitions:
    - id: "t1"
      from: "start"
      to: "end1"
      event: "go"
    - id: "t2"
      from: "start"
      to: "end2"
      event: "go"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let validator = StateMachineValidator::new(&playbook);
        let result = validator.validate();

        assert!(!result.determinism.is_deterministic);
        assert!(result.issues.iter().any(
            |i| matches!(i, ValidationIssue::NonDeterministic { state_id, event, .. } if state_id == "start" && event == "go")
        ));
    }

    #[test]
    fn test_dot_generation() {
        let playbook = Playbook::from_yaml(VALID_PLAYBOOK).expect("parse");
        let dot = to_dot(&playbook);

        assert!(dot.contains("digraph StateMachine"));
        assert!(dot.contains("__start"));
        assert!(dot.contains("doublecircle")); // final state
        assert!(dot.contains("\"start\" -> \"middle\""));
    }
}
