//! Monte Carlo Tree Search implementation for code generation.
//!
//! # Overview
//!
//! MCTS is a heuristic search algorithm that builds a search tree iteratively
//! through four phases: Selection, Expansion, Simulation, and Backpropagation.
//!
//! For code generation:
//! - **State**: Partial AST (Abstract Syntax Tree)
//! - **Action**: Transform rules (e.g., add statement, wrap in loop)
//! - **Reward**: Compilation success (binary: 0 or 1)
//!
//! # Example
//!
//! ```rust
//! use entrenar::search::{MctsSearch, MctsConfig, State, Action};
//!
//! // Define your state and action spaces
//! #[derive(Clone, PartialEq, Eq, Hash)]
//! struct CodeState {
//!     ast_tokens: Vec<String>,
//! }
//!
//! impl State for CodeState {
//!     fn is_terminal(&self) -> bool {
//!         self.ast_tokens.iter().any(|t| t == "EOF")
//!     }
//! }
//!
//! // Create MCTS searcher with default config
//! let config = MctsConfig::default();
//! // let mcts = MctsSearch::new(initial_state, action_space, config);
//! ```

mod config;
mod node;
mod search;
mod traits;
mod tree;

// Re-export all public types
pub use config::MctsConfig;
pub use node::{Node, NodeId, NodeStats};
pub use search::{MctsResult, MctsSearch, MctsStats};
pub use traits::{Action, ActionSpace, PolicyNetwork, State, StateSpace};
pub use tree::SearchTree;

/// Type alias for reward values (typically 0.0 to 1.0)
pub type Reward = f64;
