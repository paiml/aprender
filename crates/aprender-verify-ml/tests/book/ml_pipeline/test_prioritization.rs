//! Test prioritization examples from the book

use verificar::data::CodeFeatures;
use verificar::ml::RLTestPrioritizer;

#[test]
fn test_basic_prioritization_example() {
    // Example: Basic test prioritization
    let prioritizer = RLTestPrioritizer::new();

    let features = vec![
        CodeFeatures {
            ast_depth: 5,
            num_operators: 10,
            cyclomatic_complexity: 3.0,
            ..Default::default()
        },
        CodeFeatures {
            ast_depth: 15,
            num_operators: 50,
            cyclomatic_complexity: 20.0,
            uses_edge_values: true,
            ..Default::default()
        },
    ];

    let priorities = prioritizer.prioritize(&features);

    assert_eq!(priorities.len(), 2);
    // Thompson Sampling is random - just verify both indices are present
    assert!(priorities.contains(&0));
    assert!(priorities.contains(&1));
}

#[test]
fn test_rl_feedback_example() {
    // Example: Learning from test results
    let mut prioritizer = RLTestPrioritizer::new();

    let features = CodeFeatures {
        ast_depth: 10,
        num_operators: 30,
        cyclomatic_complexity: 8.0,
        ..Default::default()
    };

    // Update with test result
    prioritizer.update_feedback(&features, true); // Test found a bug

    // Prioritizer should learn that similar code is more likely to have bugs
    let similar_features = CodeFeatures {
        ast_depth: 11,
        num_operators: 32,
        cyclomatic_complexity: 9.0,
        ..Default::default()
    };

    let priorities = prioritizer.prioritize(&[similar_features]);
    assert!(!priorities.is_empty());
}

#[test]
fn test_exploration_vs_exploitation_example() {
    // Example: Balancing exploration and exploitation
    let mut prioritizer = RLTestPrioritizer::new();

    // Train on some known patterns
    for _ in 0..10 {
        let features = CodeFeatures {
            ast_depth: 5,
            num_operators: 10,
            cyclomatic_complexity: 3.0,
            ..Default::default()
        };
        prioritizer.update_feedback(&features, false); // Simple code rarely has bugs
    }

    // Prioritizer should still explore new patterns
    let unexplored = CodeFeatures {
        ast_depth: 20,
        num_operators: 100,
        cyclomatic_complexity: 30.0,
        uses_edge_values: true,
        ..Default::default()
    };

    let priorities = prioritizer.prioritize(&[unexplored]);
    assert_eq!(priorities.len(), 1);
}

#[test]
fn test_batch_prioritization_example() {
    // Example: Prioritizing a large batch of tests
    let prioritizer = RLTestPrioritizer::new();

    let features: Vec<CodeFeatures> = (0..100)
        .map(|i| CodeFeatures {
            ast_depth: i % 20,
            num_operators: i * 3,
            cyclomatic_complexity: (i as f32) * 0.5,
            uses_edge_values: i % 10 == 0,
            ..Default::default()
        })
        .collect();

    let priorities = prioritizer.prioritize(&features);

    assert_eq!(priorities.len(), 100);
    // All indices should be unique
    let mut sorted = priorities.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 100);
}
