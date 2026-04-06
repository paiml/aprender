use super::*;
use proptest::prelude::*;

proptest! {
    /// Token IDs are always unique and monotonically increasing
    #[test]
    fn token_ids_unique(count in 1usize..100) {
        let tokens: Vec<Token> = (0..count).map(|_| Token::new()).collect();
        let ids: std::collections::HashSet<u64> = tokens.iter().map(|t| t.id()).collect();
        prop_assert_eq!(ids.len(), tokens.len());
    }

    /// join_tokens always produces a new token with a unique ID
    #[test]
    fn join_produces_unique_token(count in 0usize..20) {
        let tokens: Vec<Token> = (0..count).map(|_| Token::new()).collect();
        let joined = join_tokens(&tokens);
        for t in &tokens {
            prop_assert_ne!(joined.id(), t.id());
        }
    }

    /// Token graph without cycles is always cycle-free
    #[test]
    fn linear_graph_has_no_cycle(count in 2usize..20) {
        let mut graph = TokenGraph::new();
        let tokens: Vec<Token> = (0..count).map(|_| Token::new()).collect();

        for t in &tokens {
            graph.create_token(*t);
        }

        // Create linear chain: t0 <- t1 <- t2 <- ...
        for i in 1..tokens.len() {
            graph.add_dependency(tokens[i], tokens[i - 1]);
        }

        prop_assert!(!graph.has_cycle());
    }

    /// Memory ordering conversions are consistent
    #[test]
    fn memory_ordering_ptx_modifiers_nonempty(_dummy in 0u8..4) {
        let orderings = [
            MemoryOrdering::Weak,
            MemoryOrdering::Relaxed,
            MemoryOrdering::Acquire,
            MemoryOrdering::Release,
        ];
        for ordering in orderings {
            let modifier = ordering.to_ptx_modifier();
            prop_assert!(!modifier.is_empty());
            prop_assert!(modifier.starts_with('.'));
        }
    }

    /// Memory scope conversions are consistent
    #[test]
    fn memory_scope_ptx_scopes_nonempty(_dummy in 0u8..5) {
        let scopes = [
            MemoryScope::Thread,
            MemoryScope::Block,
            MemoryScope::Cluster,
            MemoryScope::Device,
            MemoryScope::System,
        ];
        for scope in scopes {
            let ptx_scope = scope.to_ptx_scope();
            prop_assert!(!ptx_scope.is_empty());
            prop_assert!(ptx_scope.starts_with('.'));
        }
    }

    /// Token::from_id preserves the ID
    #[test]
    fn token_from_id_preserves(id in 1u64..u64::MAX) {
        let t = Token::from_id(id);
        prop_assert_eq!(t.id(), id);
    }
}
