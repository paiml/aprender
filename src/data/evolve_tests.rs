//! Tests for EvolKit-style instruction evolution (GH-453)

use super::*;

#[test]
fn test_evolve_single_round() {
    let config = EvolConfig::default();
    let results = evolve_instruction("Write a sorting algorithm", &config);
    assert_eq!(results.len(), 1);
    assert!(results[0].instruction.len() > "Write a sorting algorithm".len());
    assert_eq!(results[0].round, 0);
}

#[test]
fn test_evolve_multi_round() {
    let config = EvolConfig::default().with_rounds(3);
    let results = evolve_instruction("Implement a stack", &config);
    assert_eq!(results.len(), 3);
    for (i, r) in results.iter().enumerate() {
        assert_eq!(r.round, i);
    }
    // Each round should produce progressively longer text
    assert!(results[2].instruction.len() > results[0].instruction.len());
}

#[test]
fn test_evolve_deterministic() {
    let config = EvolConfig::default().with_seed(123).with_rounds(2);
    let r1 = evolve_instruction("Build a cache", &config);
    let r2 = evolve_instruction("Build a cache", &config);
    assert_eq!(r1, r2);
}

#[test]
fn test_evolve_different_seeds() {
    let c1 = EvolConfig::default().with_seed(1);
    let c2 = EvolConfig::default().with_seed(2);
    let r1 = evolve_instruction("Build a cache", &c1);
    let r2 = evolve_instruction("Build a cache", &c2);
    // Different seeds produce different evolutions
    assert_ne!(r1[0].instruction, r2[0].instruction);
}

#[test]
fn test_evolve_batch() {
    let config = EvolConfig::default().with_rounds(2);
    let instructions = vec![
        "Write a sort".to_string(),
        "Build a tree".to_string(),
        "Parse JSON".to_string(),
    ];
    let results = evolve_batch(&instructions, &config);
    assert_eq!(results.len(), 6); // 3 instructions × 2 rounds
}

#[test]
fn test_evolve_batch_empty() {
    let config = EvolConfig::default();
    let results = evolve_batch(&[], &config);
    assert!(results.is_empty());
}

#[test]
fn test_strategy_add_constraints() {
    let config = EvolConfig::default().with_strategies(vec![EvolStrategy::AddConstraints]);
    let results = evolve_instruction("Write a function", &config);
    // Should contain a constraint clause
    assert!(results[0].instruction.contains("Write a function, "));
    assert_eq!(results[0].strategy, EvolStrategy::AddConstraints);
}

#[test]
fn test_strategy_deepen_reasoning() {
    let config = EvolConfig::default().with_strategies(vec![EvolStrategy::DeepenReasoning]);
    let results = evolve_instruction("Solve this problem", &config);
    assert!(results[0].instruction.contains("reasoning step by step"));
}

#[test]
fn test_strategy_concretize() {
    let config = EvolConfig::default().with_strategies(vec![EvolStrategy::Concretize]);
    let results = evolve_instruction("Design a system", &config);
    assert!(results[0].instruction.contains("context of"));
}

#[test]
fn test_strategy_increase_complexity() {
    let config = EvolConfig::default().with_strategies(vec![EvolStrategy::IncreaseComplexity]);
    let results = evolve_instruction("Build a server", &config);
    assert!(results[0].instruction.len() > "Build a server".len());
}

#[test]
fn test_strategy_breadth_mutation() {
    let config = EvolConfig::default().with_strategies(vec![EvolStrategy::BreadthMutation]);
    let results = evolve_instruction("Implement caching", &config);
    assert!(results[0].instruction.contains("Implement caching"));
    assert_eq!(results[0].strategy, EvolStrategy::BreadthMutation);
}

#[test]
fn test_source_hash_provenance() {
    let config = EvolConfig::default();
    let r1 = evolve_instruction("Task A", &config);
    let r2 = evolve_instruction("Task B", &config);
    assert_ne!(r1[0].source_hash, r2[0].source_hash);

    // Same input always produces same hash
    let r3 = evolve_instruction("Task A", &config);
    assert_eq!(r1[0].source_hash, r3[0].source_hash);
}

#[test]
fn test_config_default() {
    let config = EvolConfig::default();
    assert_eq!(config.rounds, 1);
    assert_eq!(config.strategies.len(), 4);
    assert_eq!(config.seed, 42);
}

#[test]
fn test_config_with_rounds_min_one() {
    let config = EvolConfig::default().with_rounds(0);
    assert_eq!(config.rounds, 1);
}

#[test]
fn test_config_with_empty_strategies_preserved() {
    let config = EvolConfig::default().with_strategies(vec![]);
    // Empty strategies should preserve existing (non-empty)
    assert_eq!(config.strategies.len(), 4);
}

/// FALSIFY-EVOL-001: Evolution always produces longer output
#[test]
fn falsify_evol_001_length_increase() {
    let config = EvolConfig::default().with_rounds(3);
    let inputs = [
        "Sort an array",
        "Build a tree",
        "Parse a file",
        "Send a request",
        "Handle errors",
    ];
    for input in &inputs {
        let results = evolve_instruction(input, &config);
        assert!(
            results.last().unwrap().instruction.len() > input.len(),
            "Evolution did not increase length for: {input}"
        );
    }
}

/// FALSIFY-EVOL-002: Batch preserves all inputs
#[test]
fn falsify_evol_002_batch_preserves_all() {
    let config = EvolConfig::default();
    let instructions: Vec<String> = (0..10).map(|i| format!("Task {i}")).collect();
    let results = evolve_batch(&instructions, &config);
    assert_eq!(results.len(), instructions.len());

    // Each result should reference a different source hash
    let hashes: HashSet<u64> = results.iter().map(|r| r.source_hash).collect();
    assert_eq!(hashes.len(), instructions.len());
}

#[test]
fn test_evolved_instruction_debug() {
    let config = EvolConfig::default();
    let results = evolve_instruction("Test", &config);
    let debug = format!("{:?}", results[0]);
    assert!(debug.contains("EvolvedInstruction"));
}

use std::collections::HashSet;
