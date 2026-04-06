//! TensorLogic: Neuro-Symbolic Reasoning via Tensor Operations
//!
//! This module implements the TensorLogic paradigm (Domingos, 2025), unifying neural
//! and symbolic reasoning through tensor operations. All logical operations are
//! expressed as Einstein summations, enabling:
//!
//! - **Differentiable inference**: Backpropagation through logical reasoning
//! - **Dual-mode operation**: Boolean (guaranteed correctness) or Continuous (learnable)
//! - **Knowledge graph reasoning**: RESCAL factorization and embedding space queries
//!
//! # Toyota Way Principles
//!
//! - **Jidoka**: Boolean mode guarantees no hallucinations (output ⊆ derivable facts)
//! - **Poka-Yoke**: Type-safe mode selection prevents accidental mixing
//! - **Genchi Genbutsu**: Explicit tensor equations for auditability
//!
//! # Example
//!
//! ```rust
//! use aprender::logic::{LogicMode, logical_join, logical_project};
//!
//! // Family tree reasoning: Grandparent = Parent @ Parent
//! let parent = vec![
//!     vec![0.0, 1.0, 0.0],  // Alice is parent of Bob
//!     vec![0.0, 0.0, 1.0],  // Bob is parent of Charlie
//!     vec![0.0, 0.0, 0.0],  // Charlie has no children
//! ];
//!
//! let grandparent = logical_join(&parent, &parent, LogicMode::Boolean);
//! // grandparent[0][2] = 1.0 (Alice is grandparent of Charlie)
//! ```
//!
//! # References
//!
//! - Domingos, P. (2025). "Tensor Logic: The Language of AI." arXiv:2510.12269
//! - Nickel, M. et al. (2011). "RESCAL: A Three-Way Model for Collective Learning"
//! - Bordes, A. et al. (2013). "TransE: Translating Embeddings for Multi-relational Data"

mod embed;
mod ops;
mod program;

pub use ops::{
    apply_nonlinearity, apply_nonlinearity_with_mask, apply_nonlinearity_with_temperature,
    logical_join, logical_negation, logical_project, logical_select, logical_union, LogicMode,
    Nonlinearity,
};

pub use program::{Equation, ProgramBuilder, TensorProgram};

pub use embed::{BilinearScorer, EmbeddingSpace, RelationMatrix, RescalFactorizer};

#[cfg(test)]
#[path = "logic_tests.rs"]
mod tests;
