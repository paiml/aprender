//! EvolKit-style instruction evolution (GH-453)
//!
//! Implements WizardLM's Evol-Instruct method for making instructions more
//! complex (depth evolution) and more diverse (breadth evolution).
//! Pure text transformation — no model required.
//!
//! Reference: Xu et al. (2023). WizardLM: Empowering Large Language Models
//! to Follow Complex Instructions. arXiv:2304.12244.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Evolution strategy for instruction complexity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvolStrategy {
    /// Add constraints ("do X without using Y")
    AddConstraints,
    /// Deepen reasoning ("explain step by step why...")
    DeepenReasoning,
    /// Concretize ("give a specific example of...")
    Concretize,
    /// Increase complexity ("now consider the case where...")
    IncreaseComplexity,
    /// Breadth mutation ("rephrase as a different task type")
    BreadthMutation,
}

/// Configuration for instruction evolution.
#[derive(Debug, Clone)]
pub struct EvolConfig {
    /// Number of evolution rounds per instruction
    pub rounds: usize,
    /// Strategies to apply (cycled through rounds)
    pub strategies: Vec<EvolStrategy>,
    /// Seed for deterministic evolution
    pub seed: u64,
    /// Minimum output length increase factor (1.0 = no minimum)
    pub min_length_factor: f32,
}

impl Default for EvolConfig {
    fn default() -> Self {
        Self {
            rounds: 1,
            strategies: vec![
                EvolStrategy::AddConstraints,
                EvolStrategy::DeepenReasoning,
                EvolStrategy::Concretize,
                EvolStrategy::IncreaseComplexity,
            ],
            seed: 42,
            min_length_factor: 1.0,
        }
    }
}

impl EvolConfig {
    /// Create config with specified rounds and all strategies.
    #[must_use]
    pub fn with_rounds(mut self, rounds: usize) -> Self {
        self.rounds = rounds.max(1);
        self
    }

    /// Set the evolution strategies.
    #[must_use]
    pub fn with_strategies(mut self, strategies: Vec<EvolStrategy>) -> Self {
        if !strategies.is_empty() {
            self.strategies = strategies;
        }
        self
    }

    /// Set the seed for deterministic evolution.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// An evolved instruction with provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvolvedInstruction {
    /// The evolved instruction text
    pub instruction: String,
    /// Strategy used for this evolution
    pub strategy: EvolStrategy,
    /// Evolution round (0-indexed)
    pub round: usize,
    /// Original instruction hash for provenance
    pub source_hash: u64,
}

/// Evolve a single instruction through configured strategies.
///
/// Returns all intermediate evolved versions (one per round).
#[must_use]
pub fn evolve_instruction(instruction: &str, config: &EvolConfig) -> Vec<EvolvedInstruction> {
    let source_hash = hash_str(instruction);
    let mut results = Vec::with_capacity(config.rounds);
    let mut current = instruction.to_string();

    for round in 0..config.rounds {
        let strategy_idx = (round + (config.seed as usize)) % config.strategies.len();
        let strategy = config.strategies[strategy_idx];
        current = apply_strategy(&current, strategy, config.seed.wrapping_add(round as u64));

        results.push(EvolvedInstruction {
            instruction: current.clone(),
            strategy,
            round,
            source_hash,
        });
    }

    results
}

/// Evolve a batch of instructions.
///
/// Returns flattened list of all evolved instructions.
#[must_use]
pub fn evolve_batch(instructions: &[String], config: &EvolConfig) -> Vec<EvolvedInstruction> {
    instructions
        .iter()
        .flat_map(|inst| evolve_instruction(inst, config))
        .collect()
}

/// Apply a single evolution strategy to an instruction.
fn apply_strategy(instruction: &str, strategy: EvolStrategy, seed: u64) -> String {
    match strategy {
        EvolStrategy::AddConstraints => add_constraints(instruction, seed),
        EvolStrategy::DeepenReasoning => deepen_reasoning(instruction),
        EvolStrategy::Concretize => concretize(instruction, seed),
        EvolStrategy::IncreaseComplexity => increase_complexity(instruction, seed),
        EvolStrategy::BreadthMutation => breadth_mutation(instruction, seed),
    }
}

/// Add constraints to make the instruction more specific.
fn add_constraints(instruction: &str, seed: u64) -> String {
    let constraints = [
        "without using any external libraries",
        "in under 50 lines of code",
        "ensuring O(n) time complexity",
        "handling all edge cases including empty input",
        "with comprehensive error handling",
        "using only the standard library",
        "optimized for memory efficiency",
        "with thread safety guarantees",
    ];
    let idx = (seed as usize) % constraints.len();
    format!("{instruction}, {}", constraints[idx])
}

/// Deepen the reasoning required.
fn deepen_reasoning(instruction: &str) -> String {
    format!(
        "{instruction}. Explain your reasoning step by step, \
         including why alternative approaches were rejected."
    )
}

/// Make the instruction more concrete with specific examples.
fn concretize(instruction: &str, seed: u64) -> String {
    let domains = [
        "a web server handling concurrent requests",
        "a financial transaction processing system",
        "a real-time data streaming pipeline",
        "an embedded systems controller",
        "a machine learning inference engine",
        "a distributed key-value store",
    ];
    let idx = (seed as usize) % domains.len();
    format!(
        "{instruction}. Apply this specifically to the context of {}.",
        domains[idx]
    )
}

/// Increase the complexity of the instruction.
fn increase_complexity(instruction: &str, seed: u64) -> String {
    let extensions = [
        "Now extend this to handle multiple concurrent users",
        "Add support for graceful degradation under load",
        "Include a caching layer with configurable eviction",
        "Add observability with structured logging and metrics",
        "Support both synchronous and asynchronous execution modes",
        "Handle partial failures with retry and circuit breaker patterns",
    ];
    let idx = (seed as usize) % extensions.len();
    format!("{instruction}. {}", extensions[idx])
}

/// Mutate the instruction to a different task type.
fn breadth_mutation(instruction: &str, seed: u64) -> String {
    let mutations = [
        "Rewrite this as a debugging exercise: given the following buggy implementation, identify and fix the issues:",
        "Convert this into a code review task: review the following implementation and suggest improvements:",
        "Transform this into a testing task: write comprehensive tests for:",
        "Reframe as an optimization challenge: profile and optimize the performance of:",
        "Convert to a documentation task: write clear API documentation for:",
    ];
    let idx = (seed as usize) % mutations.len();
    format!("{} {instruction}", mutations[idx])
}

fn hash_str(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
#[path = "evolve_tests.rs"]
mod tests;
