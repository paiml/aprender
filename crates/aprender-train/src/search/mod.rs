//! MCTS (Monte Carlo Tree Search) for Code Generation
//!
//! This module implements MCTS for code translation search spaces where:
//! - State: Partial AST representation
//! - Action: Transform rules (AST transformations)
//! - Reward: Compilation success (0 or 1)
//!
//! Uses UCB1/UCT selection policy for balancing exploration/exploitation,
//! with optional policy network guidance via `aprender`.

pub mod mcts;

pub use mcts::{
    Action, ActionSpace, MctsConfig, MctsResult, MctsSearch, MctsStats, Node, NodeId, NodeStats,
    PolicyNetwork, Reward, SearchTree, State, StateSpace,
};
