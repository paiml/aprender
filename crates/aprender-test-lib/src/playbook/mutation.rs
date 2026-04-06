//! Mutation testing support for playbook validation.
//!
//! Implements M1-M5 mutation classes for falsification protocol.
//! Reference: Fabbri et al., "Mutation Testing Applied to Validate
//! Specifications Based on Statecharts" (ISSRE 1999)

use super::schema::Playbook;
use std::collections::HashMap;

/// Mutation classes for state machine testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MutationClass {
    /// M1: State removal - remove a state
    StateRemoval,
    /// M2: Transition removal - remove a transition
    TransitionRemoval,
    /// M3: Event swap - swap event triggers between transitions
    EventSwap,
    /// M4: Target swap - change transition target to different state
    TargetSwap,
    /// M5: Guard negation - negate guard conditions
    GuardNegation,
}

impl MutationClass {
    /// Get all mutation classes.
    pub fn all() -> Vec<MutationClass> {
        vec![
            MutationClass::StateRemoval,
            MutationClass::TransitionRemoval,
            MutationClass::EventSwap,
            MutationClass::TargetSwap,
            MutationClass::GuardNegation,
        ]
    }

    /// Get the mutation class identifier (M1-M5).
    pub fn id(&self) -> &'static str {
        match self {
            MutationClass::StateRemoval => "M1",
            MutationClass::TransitionRemoval => "M2",
            MutationClass::EventSwap => "M3",
            MutationClass::TargetSwap => "M4",
            MutationClass::GuardNegation => "M5",
        }
    }

    /// Get a description of the mutation class.
    pub fn description(&self) -> &'static str {
        match self {
            MutationClass::StateRemoval => "Remove a state from the machine",
            MutationClass::TransitionRemoval => "Remove a transition from the machine",
            MutationClass::EventSwap => "Swap event triggers between two transitions",
            MutationClass::TargetSwap => "Change a transition's target to a different state",
            MutationClass::GuardNegation => "Negate a transition's guard condition",
        }
    }
}

/// A mutant is a modified version of the original playbook.
#[derive(Debug, Clone)]
pub struct Mutant {
    /// Unique identifier for this mutant
    pub id: String,
    /// Mutation class applied
    pub class: MutationClass,
    /// Description of the mutation
    pub description: String,
    /// The mutated playbook
    pub playbook: Playbook,
}

/// Result of running tests against a mutant.
#[derive(Debug, Clone)]
pub struct MutantResult {
    /// Mutant identifier
    pub mutant_id: String,
    /// Mutation class
    pub class: MutationClass,
    /// Whether the mutant was killed (test detected the mutation)
    pub killed: bool,
    /// How the mutant was killed (if killed)
    pub kill_reason: Option<String>,
}

/// Mutation score summary.
#[derive(Debug, Clone)]
pub struct MutationScore {
    /// Total mutants generated
    pub total_mutants: usize,
    /// Mutants killed by tests
    pub killed: usize,
    /// Mutants that survived
    pub survived: usize,
    /// Mutation score (killed / total)
    pub score: f64,
    /// Results by mutation class
    pub by_class: HashMap<MutationClass, ClassScore>,
}

/// Score for a single mutation class.
#[derive(Debug, Clone, Default)]
pub struct ClassScore {
    pub total: usize,
    pub killed: usize,
    pub score: f64,
}

/// Mutation generator for playbooks.
pub struct MutationGenerator<'a> {
    playbook: &'a Playbook,
}

impl<'a> MutationGenerator<'a> {
    /// Create a new mutation generator for the given playbook.
    pub fn new(playbook: &'a Playbook) -> Self {
        Self { playbook }
    }

    /// Generate all possible mutants across all mutation classes.
    pub fn generate_all(&self) -> Vec<Mutant> {
        let mut mutants = Vec::new();
        mutants.extend(self.generate_state_removals());
        mutants.extend(self.generate_transition_removals());
        mutants.extend(self.generate_event_swaps());
        mutants.extend(self.generate_target_swaps());
        mutants.extend(self.generate_guard_negations());
        mutants
    }

    /// Generate mutants for a specific class.
    pub fn generate(&self, class: MutationClass) -> Vec<Mutant> {
        match class {
            MutationClass::StateRemoval => self.generate_state_removals(),
            MutationClass::TransitionRemoval => self.generate_transition_removals(),
            MutationClass::EventSwap => self.generate_event_swaps(),
            MutationClass::TargetSwap => self.generate_target_swaps(),
            MutationClass::GuardNegation => self.generate_guard_negations(),
        }
    }

    /// M1: Generate state removal mutants.
    fn generate_state_removals(&self) -> Vec<Mutant> {
        let mut mutants = Vec::new();

        for state_id in self.playbook.machine.states.keys() {
            // Skip initial state (would make playbook invalid)
            if *state_id == self.playbook.machine.initial {
                continue;
            }

            let mut mutated = self.playbook.clone();
            mutated.machine.states.remove(state_id);

            // Also remove transitions involving this state
            mutated
                .machine
                .transitions
                .retain(|t| t.from != *state_id && t.to != *state_id);

            mutants.push(Mutant {
                id: format!("M1_{}", state_id),
                class: MutationClass::StateRemoval,
                description: format!("Remove state '{}'", state_id),
                playbook: mutated,
            });
        }

        mutants
    }

    /// M2: Generate transition removal mutants.
    fn generate_transition_removals(&self) -> Vec<Mutant> {
        let mut mutants = Vec::new();

        for (idx, transition) in self.playbook.machine.transitions.iter().enumerate() {
            let mut mutated = self.playbook.clone();
            mutated.machine.transitions.remove(idx);

            // Only generate if still valid (at least one transition remains)
            if !mutated.machine.transitions.is_empty() {
                mutants.push(Mutant {
                    id: format!("M2_{}", transition.id),
                    class: MutationClass::TransitionRemoval,
                    description: format!("Remove transition '{}'", transition.id),
                    playbook: mutated,
                });
            }
        }

        mutants
    }

    /// M3: Generate event swap mutants.
    fn generate_event_swaps(&self) -> Vec<Mutant> {
        let mut mutants = Vec::new();
        let transitions = &self.playbook.machine.transitions;

        for i in 0..transitions.len() {
            for j in (i + 1)..transitions.len() {
                // Only swap if events are different
                if transitions[i].event != transitions[j].event {
                    let mut mutated = self.playbook.clone();

                    // Swap events
                    let event_i = transitions[i].event.clone();
                    let event_j = transitions[j].event.clone();
                    mutated.machine.transitions[i].event = event_j;
                    mutated.machine.transitions[j].event = event_i;

                    mutants.push(Mutant {
                        id: format!("M3_{}_{}", transitions[i].id, transitions[j].id),
                        class: MutationClass::EventSwap,
                        description: format!(
                            "Swap events between '{}' and '{}'",
                            transitions[i].id, transitions[j].id
                        ),
                        playbook: mutated,
                    });
                }
            }
        }

        mutants
    }

    /// M4: Generate target swap mutants.
    fn generate_target_swaps(&self) -> Vec<Mutant> {
        let mut mutants = Vec::new();
        let state_ids: Vec<_> = self.playbook.machine.states.keys().collect();

        for (idx, transition) in self.playbook.machine.transitions.iter().enumerate() {
            for state_id in &state_ids {
                // Skip if same as original target
                if **state_id == transition.to {
                    continue;
                }

                let mut mutated = self.playbook.clone();
                mutated.machine.transitions[idx].to = (*state_id).clone();

                mutants.push(Mutant {
                    id: format!("M4_{}_{}", transition.id, state_id),
                    class: MutationClass::TargetSwap,
                    description: format!(
                        "Change target of '{}' from '{}' to '{}'",
                        transition.id, transition.to, state_id
                    ),
                    playbook: mutated,
                });
            }
        }

        mutants
    }

    /// M5: Generate guard negation mutants.
    fn generate_guard_negations(&self) -> Vec<Mutant> {
        let mut mutants = Vec::new();

        for (idx, transition) in self.playbook.machine.transitions.iter().enumerate() {
            if let Some(guard) = &transition.guard {
                let mut mutated = self.playbook.clone();

                // Negate the guard condition
                let negated = format!("!({})", guard);
                mutated.machine.transitions[idx].guard = Some(negated.clone());

                mutants.push(Mutant {
                    id: format!("M5_{}", transition.id),
                    class: MutationClass::GuardNegation,
                    description: format!(
                        "Negate guard of '{}': '{}' → '{}'",
                        transition.id, guard, negated
                    ),
                    playbook: mutated,
                });
            }
        }

        mutants
    }
}

/// Calculate mutation score from results.
pub fn calculate_mutation_score(results: &[MutantResult]) -> MutationScore {
    let total_mutants = results.len();
    let killed = results.iter().filter(|r| r.killed).count();
    let survived = total_mutants - killed;
    let score = if total_mutants > 0 {
        killed as f64 / total_mutants as f64
    } else {
        1.0
    };

    // Calculate per-class scores
    let mut by_class: HashMap<MutationClass, ClassScore> = HashMap::new();

    for class in MutationClass::all() {
        let class_results: Vec<_> = results.iter().filter(|r| r.class == class).collect();
        let class_total = class_results.len();
        let class_killed = class_results.iter().filter(|r| r.killed).count();

        by_class.insert(
            class,
            ClassScore {
                total: class_total,
                killed: class_killed,
                score: if class_total > 0 {
                    class_killed as f64 / class_total as f64
                } else {
                    1.0
                },
            },
        );
    }

    MutationScore {
        total_mutants,
        killed,
        survived,
        score,
        by_class,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playbook::schema::Playbook;

    const TEST_PLAYBOOK: &str = r#"
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
      guard: "user.isLoggedIn"
"#;

    #[test]
    fn test_generate_state_removals() {
        let playbook = Playbook::from_yaml(TEST_PLAYBOOK).expect("parse");
        let generator = MutationGenerator::new(&playbook);
        let mutants = generator.generate(MutationClass::StateRemoval);

        // Should generate 2 mutants (middle and end, not start)
        assert_eq!(mutants.len(), 2);
        assert!(mutants
            .iter()
            .all(|m| m.class == MutationClass::StateRemoval));
    }

    #[test]
    fn test_generate_transition_removals() {
        let playbook = Playbook::from_yaml(TEST_PLAYBOOK).expect("parse");
        let generator = MutationGenerator::new(&playbook);
        let mutants = generator.generate(MutationClass::TransitionRemoval);

        // Should generate 2 mutants (one for each transition)
        // But only 1 is valid (removing one leaves at least one)
        assert!(!mutants.is_empty());
        assert!(mutants
            .iter()
            .all(|m| m.class == MutationClass::TransitionRemoval));
    }

    #[test]
    fn test_generate_event_swaps() {
        let playbook = Playbook::from_yaml(TEST_PLAYBOOK).expect("parse");
        let generator = MutationGenerator::new(&playbook);
        let mutants = generator.generate(MutationClass::EventSwap);

        // Should generate 1 mutant (swap "next" and "finish")
        assert_eq!(mutants.len(), 1);
        assert_eq!(mutants[0].class, MutationClass::EventSwap);
    }

    #[test]
    fn test_generate_target_swaps() {
        let playbook = Playbook::from_yaml(TEST_PLAYBOOK).expect("parse");
        let generator = MutationGenerator::new(&playbook);
        let mutants = generator.generate(MutationClass::TargetSwap);

        // Each transition can target 2 other states
        assert_eq!(mutants.len(), 4);
        assert!(mutants.iter().all(|m| m.class == MutationClass::TargetSwap));
    }

    #[test]
    fn test_generate_guard_negations() {
        let playbook = Playbook::from_yaml(TEST_PLAYBOOK).expect("parse");
        let generator = MutationGenerator::new(&playbook);
        let mutants = generator.generate(MutationClass::GuardNegation);

        // Only t2 has a guard
        assert_eq!(mutants.len(), 1);
        assert_eq!(mutants[0].class, MutationClass::GuardNegation);
        assert!(mutants[0]
            .playbook
            .machine
            .transitions
            .iter()
            .any(|t| t.guard.as_deref() == Some("!(user.isLoggedIn)")));
    }

    #[test]
    fn test_generate_all() {
        let playbook = Playbook::from_yaml(TEST_PLAYBOOK).expect("parse");
        let generator = MutationGenerator::new(&playbook);
        let mutants = generator.generate_all();

        // Should have mutants from all classes
        let has_m1 = mutants
            .iter()
            .any(|m| m.class == MutationClass::StateRemoval);
        let has_m2 = mutants
            .iter()
            .any(|m| m.class == MutationClass::TransitionRemoval);
        let has_m3 = mutants.iter().any(|m| m.class == MutationClass::EventSwap);
        let has_m4 = mutants.iter().any(|m| m.class == MutationClass::TargetSwap);
        let has_m5 = mutants
            .iter()
            .any(|m| m.class == MutationClass::GuardNegation);

        assert!(has_m1);
        assert!(has_m2);
        assert!(has_m3);
        assert!(has_m4);
        assert!(has_m5);
    }

    #[test]
    fn test_calculate_mutation_score() {
        let results = vec![
            MutantResult {
                mutant_id: "M1_1".to_string(),
                class: MutationClass::StateRemoval,
                killed: true,
                kill_reason: Some("Validation failed".to_string()),
            },
            MutantResult {
                mutant_id: "M2_1".to_string(),
                class: MutationClass::TransitionRemoval,
                killed: true,
                kill_reason: Some("Test failed".to_string()),
            },
            MutantResult {
                mutant_id: "M3_1".to_string(),
                class: MutationClass::EventSwap,
                killed: false,
                kill_reason: None,
            },
        ];

        let score = calculate_mutation_score(&results);

        assert_eq!(score.total_mutants, 3);
        assert_eq!(score.killed, 2);
        assert_eq!(score.survived, 1);
        assert!((score.score - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_mutation_class_metadata() {
        assert_eq!(MutationClass::StateRemoval.id(), "M1");
        assert_eq!(MutationClass::TransitionRemoval.id(), "M2");
        assert_eq!(MutationClass::EventSwap.id(), "M3");
        assert_eq!(MutationClass::TargetSwap.id(), "M4");
        assert_eq!(MutationClass::GuardNegation.id(), "M5");
    }
}
