//! MCTS search statistics.

/// Statistics from an MCTS search
#[derive(Debug, Clone)]
pub struct MctsStats {
    /// Number of iterations performed
    pub iterations: usize,
    /// Total nodes in tree
    pub tree_size: usize,
    /// Maximum depth reached
    pub max_depth: usize,
    /// Average simulation length
    pub avg_simulation_length: f64,
    /// Root node visit count
    pub root_visits: usize,
}
