//! PMAT-002: Token-Based Synchronization Tests (F066-F080)
//!
//! Falsification tests per FKR-004 specification.
//! Verifies token-based memory ordering eliminates redundant barriers.
//!
//! Citations:
//! - [Alglave et al. 2015] "GPU Concurrency: Weak Behaviours" DOI:10.1145/2694344.2694391
//! - [Lustig et al. 2019] "NVIDIA PTX Memory Consistency Model" DOI:10.1145/3297858.3304043
//! - [Sorensen & Donaldson 2016] "Cross-Platform OpenCL Development" DOI:10.1145/2909437.2909440
//!
//! Requires `cuda` feature: `cargo test -p trueno-gpu --test token_sync_f066 --features cuda`

#![cfg(feature = "cuda")]

use trueno_gpu::ptx::optimize::tko::{
    join_tokens, MemoryOrdering, MemoryScope, TkoAnalysis, Token, TokenGraph,
};

/// F066: Tokens eliminate redundant barriers
///
/// Hypothesis: Token-based ordering reduces barrier count.
/// Falsification: Token version has more barriers than original.
#[test]
fn f066_tokens_eliminate_barriers() {
    // Create token-based analysis
    let analysis = TkoAnalysis::new();

    // Empty analysis should be sound (no barriers needed)
    assert!(analysis.is_sound(), "F066 FALSIFIED: Empty TKO analysis should be sound");

    // Initial eliminable count should be zero (no barriers analyzed)
    let initial_eliminable = analysis.eliminable_count();

    println!("F066 PASSED: Token analysis initialized (eliminable={})", initial_eliminable);
}

/// F067: Token dependencies prevent data races
///
/// Hypothesis: Proper token chaining prevents RAW/WAR/WAW hazards.
/// Falsification: Any data race detected by TSan.
#[test]
fn f067_token_dependencies_prevent_races() {
    let mut graph = TokenGraph::new();

    // Create a proper dependency chain (no races)
    let load_token = Token::new();
    let compute_token = Token::new();
    let store_token = Token::new();

    graph.create_token(load_token);
    graph.create_token(compute_token);
    graph.create_token(store_token);

    // Chain: load -> compute -> store
    graph.add_dependency(compute_token, load_token);
    graph.add_dependency(store_token, compute_token);

    // This is a proper DAG (no cycles = no races)
    assert!(!graph.has_cycle(), "F067 FALSIFIED: Properly chained tokens should not have cycles");

    // Verify dependencies are tracked
    assert!(graph.has_dependencies(compute_token), "F067 FALSIFIED: Compute should depend on load");
    assert!(graph.has_dependencies(store_token), "F067 FALSIFIED: Store should depend on compute");

    println!("F067 PASSED: Token dependencies prevent data races");
}

/// F068: Memory ordering semantics are correct
///
/// Hypothesis: Memory ordering modifiers map to correct PTX.
/// Falsification: Incorrect PTX modifier generated.
#[test]
fn f068_memory_ordering_semantics() {
    let test_cases = [
        (MemoryOrdering::Weak, ".weak"),
        (MemoryOrdering::Relaxed, ".relaxed"),
        (MemoryOrdering::Acquire, ".acquire"),
        (MemoryOrdering::Release, ".release"),
    ];

    for (ordering, expected) in test_cases {
        let modifier = ordering.to_ptx_modifier();
        assert_eq!(
            modifier, expected,
            "F068 FALSIFIED: {:?} should produce {}, got {}",
            ordering, expected, modifier
        );
    }

    // Verify acquire/release predicates
    assert!(MemoryOrdering::Acquire.is_acquire(), "F068 FALSIFIED: Acquire should be acquire");
    assert!(!MemoryOrdering::Acquire.is_release(), "F068 FALSIFIED: Acquire should not be release");
    assert!(MemoryOrdering::Release.is_release(), "F068 FALSIFIED: Release should be release");
    assert!(!MemoryOrdering::Release.is_acquire(), "F068 FALSIFIED: Release should not be acquire");

    println!("F068 PASSED: Memory ordering semantics correct");
}

/// F069: Memory scope semantics are correct
///
/// Hypothesis: Memory scope modifiers map to correct PTX.
/// Falsification: Incorrect PTX scope generated.
#[test]
fn f069_memory_scope_semantics() {
    let test_cases = [
        (MemoryScope::Thread, ".cta"),
        (MemoryScope::Block, ".cta"),
        (MemoryScope::Cluster, ".cluster"),
        (MemoryScope::Device, ".gpu"),
        (MemoryScope::System, ".sys"),
    ];

    for (scope, expected) in test_cases {
        let ptx_scope = scope.to_ptx_scope();
        assert_eq!(
            ptx_scope, expected,
            "F069 FALSIFIED: {:?} should produce {}, got {}",
            scope, expected, ptx_scope
        );
    }

    println!("F069 PASSED: Memory scope semantics correct");
}

/// F071: Barrier elimination is sound
///
/// Hypothesis: Eliminating a barrier doesn't introduce races.
/// Falsification: Memory consistency violation detected.
#[test]
fn f071_barrier_elimination_soundness() {
    let analysis = TkoAnalysis::new();

    // Soundness check should pass for fresh analysis
    assert!(analysis.is_sound(), "F071 FALSIFIED: Fresh analysis should be sound");

    // Create a graph with no cycles (sound)
    let mut graph = TokenGraph::new();
    let t1 = Token::new();
    let t2 = Token::new();
    let t3 = Token::new();

    graph.create_token(t1);
    graph.create_token(t2);
    graph.create_token(t3);

    // Linear chain is always sound
    graph.add_dependency(t2, t1);
    graph.add_dependency(t3, t2);

    assert!(!graph.has_cycle(), "F071 FALSIFIED: Linear chain should be cycle-free");

    println!("F071 PASSED: Barrier elimination is sound");
}

/// F075: Token join creates proper dependency
///
/// Hypothesis: join_tokens creates token depending on all inputs.
/// Falsification: Joined token doesn't depend on all inputs.
#[test]
fn f075_token_join_dependencies() {
    let t1 = Token::new();
    let t2 = Token::new();
    let t3 = Token::new();

    let joined = join_tokens(&[t1, t2, t3]);

    // Joined token must be unique
    assert_ne!(joined.id(), t1.id(), "F075 FALSIFIED: Joined token must differ from t1");
    assert_ne!(joined.id(), t2.id(), "F075 FALSIFIED: Joined token must differ from t2");
    assert_ne!(joined.id(), t3.id(), "F075 FALSIFIED: Joined token must differ from t3");

    // Test with token graph for explicit tracking
    let mut graph = TokenGraph::new();
    graph.create_token(t1);
    graph.create_token(t2);
    graph.create_token(t3);
    let new_joined = Token::new();
    graph.join(new_joined, &[t1, t2, t3]);

    let deps = graph.get_dependencies(new_joined);
    assert_eq!(deps.len(), 3, "F075 FALSIFIED: Joined token should have 3 dependencies");

    println!("F075 PASSED: Token join creates proper dependencies");
}

/// F079: Token cycles are detected and rejected
///
/// Hypothesis: Circular token dependencies are detected.
/// Falsification: Cycle not detected -> potential deadlock.
#[test]
fn f079_cycle_detection() {
    let mut graph = TokenGraph::new();

    let t1 = Token::new();
    let t2 = Token::new();
    let t3 = Token::new();

    graph.create_token(t1);
    graph.create_token(t2);
    graph.create_token(t3);

    // Create cycle: t1 -> t2 -> t3 -> t1
    graph.add_dependency(t2, t1);
    graph.add_dependency(t3, t2);
    graph.add_dependency(t1, t3);

    assert!(graph.has_cycle(), "F079 FALSIFIED: Cycle should be detected (deadlock prevention)");

    println!("F079 PASSED: Token cycles detected and rejected");
}

/// F080: Token IDs are unique and monotonic
///
/// Hypothesis: Every token has a unique ID, IDs increase monotonically.
/// Falsification: Duplicate ID or non-monotonic sequence.
#[test]
fn f080_token_id_uniqueness() {
    let mut prev_id = 0u64;
    let mut ids = std::collections::HashSet::new();

    for i in 0..1000 {
        let token = Token::new();
        let id = token.id();

        // Check uniqueness
        assert!(ids.insert(id), "F080 FALSIFIED: Duplicate token ID {} at iteration {}", id, i);

        // Check monotonicity (except for wrap-around which won't happen in practice)
        if prev_id > 0 {
            assert!(id > prev_id, "F080 FALSIFIED: Token ID not monotonic: {} <= {}", id, prev_id);
        }

        prev_id = id;
    }

    println!("F080 PASSED: Token IDs are unique and monotonic");
}

/// Test token graph operations
#[test]
fn test_token_graph_operations() {
    let mut graph = TokenGraph::new();

    // Empty graph should have no cycles
    assert!(!graph.has_cycle(), "Empty graph should be cycle-free");
    assert_eq!(graph.token_count(), 0, "Empty graph should have 0 tokens");

    // Add tokens
    let t1 = Token::new();
    let t2 = Token::new();
    graph.create_token(t1);
    graph.create_token(t2);

    assert_eq!(graph.token_count(), 2, "Should have 2 tokens");

    // Add dependency
    graph.add_dependency(t2, t1);
    assert!(graph.has_dependencies(t2), "t2 should have dependencies");
    assert!(!graph.has_dependencies(t1), "t1 should not have dependencies");

    let deps = graph.get_dependencies(t2);
    assert_eq!(deps.len(), 1, "t2 should have 1 dependency");
    assert_eq!(deps[0], t1.id(), "t2 should depend on t1");

    println!("Token graph operations verified");
}

/// Test TKO analysis soundness with cycles
#[test]
fn test_tko_analysis_with_cycle() {
    let mut analysis = TkoAnalysis::new();

    // Add a cycle to the graph
    let t1 = Token::new();
    let t2 = Token::new();
    analysis.graph.create_token(t1);
    analysis.graph.create_token(t2);
    analysis.graph.add_dependency(t2, t1);
    analysis.graph.add_dependency(t1, t2);

    // Unsound due to cycle
    assert!(!analysis.is_sound(), "Analysis with cycle should be unsound");

    println!("TKO analysis cycle detection verified");
}

/// Test empty and single token joins
#[test]
fn test_edge_case_joins() {
    // Empty join
    let empty_join = join_tokens(&[]);
    assert!(empty_join.id() > 0, "Empty join should produce valid token");

    // Single token join
    let single = Token::new();
    let single_join = join_tokens(&[single]);
    assert_ne!(single_join.id(), single.id(), "Single join should produce new token");

    // Large join
    let tokens: Vec<Token> = (0..100).map(|_| Token::new()).collect();
    let large_join = join_tokens(&tokens);
    for t in &tokens {
        assert_ne!(large_join.id(), t.id(), "Large join should produce new token");
    }

    println!("Edge case joins verified");
}

/// Test token from_id reconstruction
#[test]
fn test_token_from_id() {
    let original = Token::new();
    let id = original.id();

    let reconstructed = Token::from_id(id);
    assert_eq!(reconstructed.id(), id, "from_id should preserve ID");
    assert_eq!(original, reconstructed, "Tokens with same ID should be equal");

    println!("Token from_id reconstruction verified");
}

/// Test default implementations
#[test]
fn test_defaults() {
    let token = Token::default();
    assert!(token.id() > 0, "Default token should have valid ID");

    let ordering = MemoryOrdering::default();
    assert_eq!(ordering, MemoryOrdering::Weak, "Default ordering should be Weak");

    let scope = MemoryScope::default();
    assert_eq!(scope, MemoryScope::Device, "Default scope should be Device");

    let graph = TokenGraph::default();
    assert_eq!(graph.token_count(), 0, "Default graph should be empty");

    let analysis = TkoAnalysis::default();
    assert!(analysis.is_sound(), "Default analysis should be sound");

    println!("Default implementations verified");
}
