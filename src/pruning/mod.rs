//! Neural network pruning module.
//!
//! This module provides importance scoring and sparsity mask generation
//! for neural network pruning, following the specification in
//! `docs/specifications/advanced-pruning.md`.
//!
//! # Toyota Way Principles
//! - **Jidoka**: All numerical operations validate for NaN/Inf
//! - **Poka-Yoke**: Type-safe patterns prevent invalid configurations
//! - **Genchi Genbutsu**: Calibration uses real activation data
//!
//! # Example
//!
//! ```ignore
//! use aprender::pruning::{MagnitudeImportance, Importance};
//! use aprender::nn::Linear;
//!
//! let layer = Linear::new(512, 256);
//! let importance = MagnitudeImportance::l2();
//! let scores = importance.compute(&layer, None).unwrap();
//!
//! println!("Importance stats: min={}, max={}", scores.stats.min, scores.stats.max);
//! ```
//!
//! # References
//! - Han, S., et al. (2015). Learning both weights and connections. NeurIPS.
//! - Sun, M., et al. (2023). A simple and effective pruning approach. arXiv:2306.11695.
//! - Frantar, E., & Alistarh, D. (2023). SparseGPT. ICML.
//! - Frankle, J., & Carbin, M. (2018). The Lottery Ticket Hypothesis. arXiv:1803.03635.

mod calibration;
mod depth;
mod error;
mod graph;
mod importance;
mod lottery;
mod magnitude;
mod mask;
mod pruner;
mod sparse;
mod sparsegpt;
mod wanda;
mod width;

pub use calibration::{ActivationStats, CalibrationContext};
pub use depth::{BlockImportanceScores, DepthPruner, DepthPruningResult};
pub use error::PruningError;
pub use graph::{
    propagate_channel_pruning, DependencyGraph, DependencyType, GraphEdge, GraphNode, NodeType,
    PruningPlan,
};
pub use importance::{Importance, ImportanceScores, ImportanceStats};
pub use magnitude::{MagnitudeImportance, NormType};
pub use mask::{
    generate_block_mask, generate_column_mask, generate_nm_mask, generate_row_mask,
    generate_unstructured_mask, SparsityMask, SparsityPattern,
};
pub use pruner::{prune_module, MagnitudePruner, Pruner, PruningResult, WandaPruner};
pub use sparse::{sparsify, BlockSparseTensor, COOTensor, CSRTensor, SparseFormat, SparseTensor};
pub use sparsegpt::SparseGPTImportance;
pub use wanda::WandaImportance;
pub use width::{ChannelImportance, WidthPruner, WidthPruningResult};

// Lottery Ticket Hypothesis (Frankle & Carbin, 2018)
pub use lottery::{
    LotteryTicketConfig, LotteryTicketPruner, LotteryTicketPrunerBuilder, RewindStrategy,
    WinningTicket,
};
