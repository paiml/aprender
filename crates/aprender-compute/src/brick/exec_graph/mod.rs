//! Execution Graph and Brick Profiling Types
//!
//! This module contains types for execution path tracking and profiling:
//!
//! - **PAR-073**: BrickSample, BrickBottleneck - foundational profiling primitives
//! - **PAR-200**: BrickId, BrickCategory, SyncMode - O(1) hot path brick identification
//! - **PAR-201**: ExecutionGraph, ExecutionNode, etc. - full execution hierarchy tracking

mod node;
mod traversal;

pub use node::{
    BrickBottleneck, BrickCategory, BrickId, BrickSample, BrickStats, CategoryStats, EdgeType,
    ExecutionEdge, ExecutionNode, ExecutionNodeId, PtxRegistry, SyncMode, TransferDirection,
};
pub use traversal::ExecutionGraph;

#[cfg(test)]
mod tests;
