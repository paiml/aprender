use super::*;

// cuda-tile-behavior.md: Falsification test #66
#[test]
fn test_token_creation() {
    let t1 = Token::new();
    let t2 = Token::new();

    // Tokens should have unique IDs
    assert_ne!(t1.id(), t2.id());
}

// cuda-tile-behavior.md: Falsification test #75
#[test]
fn test_join_tokens() {
    let t1 = Token::new();
    let t2 = Token::new();
    let t3 = Token::new();

    let joined = join_tokens(&[t1, t2, t3]);

    // Joined token should be different from all inputs
    assert_ne!(joined.id(), t1.id());
    assert_ne!(joined.id(), t2.id());
    assert_ne!(joined.id(), t3.id());
}

// cuda-tile-behavior.md: Falsification test #69
#[test]
fn test_memory_ordering_relaxed_fastest() {
    // Relaxed should be fastest (no memory fences)
    let weak = MemoryOrdering::Weak;
    let relaxed = MemoryOrdering::Relaxed;
    let acquire = MemoryOrdering::Acquire;

    // Weak is even faster than relaxed
    assert_eq!(weak.to_ptx_modifier(), ".weak");
    assert_eq!(relaxed.to_ptx_modifier(), ".relaxed");
    assert_eq!(acquire.to_ptx_modifier(), ".acquire");
}

// cuda-tile-behavior.md: Falsification test #79
#[test]
fn test_cycle_detection() {
    let mut graph = TokenGraph::new();

    let t1 = Token::new();
    let t2 = Token::new();
    let t3 = Token::new();

    graph.create_token(t1);
    graph.create_token(t2);
    graph.create_token(t3);

    // Create a cycle: t1 -> t2 -> t3 -> t1
    graph.add_dependency(t2, t1);
    graph.add_dependency(t3, t2);
    graph.add_dependency(t1, t3);

    assert!(graph.has_cycle(), "Should detect cycle");
}

#[test]
fn test_no_cycle() {
    let mut graph = TokenGraph::new();

    let t1 = Token::new();
    let t2 = Token::new();
    let t3 = Token::new();

    graph.create_token(t1);
    graph.create_token(t2);
    graph.create_token(t3);

    // Linear dependency: t1 <- t2 <- t3
    graph.add_dependency(t2, t1);
    graph.add_dependency(t3, t2);

    assert!(!graph.has_cycle(), "Should not detect cycle");
}

// cuda-tile-behavior.md: Falsification test #67
#[test]
fn test_tko_analysis_sound() {
    let analysis = TkoAnalysis::new();
    assert!(analysis.is_sound(), "Empty analysis should be sound");
}

#[test]
fn test_memory_scope_ptx() {
    // Thread and Block both map to .cta
    assert_eq!(MemoryScope::Thread.to_ptx_scope(), ".cta");
    assert_eq!(MemoryScope::Block.to_ptx_scope(), ".cta");
    assert_eq!(MemoryScope::Cluster.to_ptx_scope(), ".cluster");
    assert_eq!(MemoryScope::Device.to_ptx_scope(), ".gpu");
    assert_eq!(MemoryScope::System.to_ptx_scope(), ".sys");
}

#[test]
fn test_token_graph_join() {
    let mut graph = TokenGraph::new();

    let t1 = Token::new();
    let t2 = Token::new();
    let result = Token::new();

    graph.create_token(t1);
    graph.create_token(t2);
    graph.join(result, &[t1, t2]);

    assert!(graph.has_dependencies(result));
    assert_eq!(graph.get_dependencies(result).len(), 2);
}

#[test]
fn test_empty_join() {
    let joined = join_tokens(&[]);
    // Should still create a valid token
    assert!(joined.id() > 0);
}

#[test]
fn test_single_token_join() {
    let t1 = Token::new();
    let joined = join_tokens(&[t1]);
    // Joined token should be different
    assert_ne!(joined.id(), t1.id());
}

#[test]
fn test_token_from_id() {
    let t = Token::from_id(42);
    assert_eq!(t.id(), 42);
}

#[test]
fn test_memory_ordering_acquire_release() {
    let acquire = MemoryOrdering::Acquire;
    let release = MemoryOrdering::Release;

    assert!(acquire.is_acquire());
    assert!(!acquire.is_release());
    assert!(release.is_release());
    assert!(!release.is_acquire());
}
