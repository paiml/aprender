//! MCTS configuration.
//!
//! This module contains the configuration parameters for the MCTS algorithm.

/// Configuration for MCTS search
#[derive(Debug, Clone)]
pub struct MctsConfig {
    /// Exploration constant for UCB1 (higher = more exploration)
    pub exploration_constant: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Maximum depth for simulation rollouts
    pub max_simulation_depth: usize,
    /// Whether to use policy network priors
    pub use_policy_priors: bool,
    /// Temperature for action selection (higher = more random)
    pub temperature: f64,
    /// Minimum visits before expansion
    pub min_visits_for_expansion: usize,
    /// Whether to reuse tree between searches
    pub reuse_tree: bool,
    /// Dirichlet noise alpha for exploration at root
    pub dirichlet_alpha: f64,
    /// Fraction of Dirichlet noise to add
    pub dirichlet_epsilon: f64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            exploration_constant: std::f64::consts::SQRT_2,
            max_iterations: 1000,
            max_simulation_depth: 100,
            use_policy_priors: true,
            temperature: 1.0,
            min_visits_for_expansion: 1,
            reuse_tree: false,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcts_config_default() {
        let config = MctsConfig::default();
        assert_eq!(config.max_iterations, 1000);
        assert!(config.exploration_constant > 0.0);
        assert!(config.use_policy_priors);
    }
}
