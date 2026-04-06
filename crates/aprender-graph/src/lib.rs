//! trueno-graph: GPU-first embedded graph database
//!
//! # Overview
//!
//! trueno-graph provides GPU-accelerated graph storage and algorithms for code analysis.
//! Built on PAIML's proven infrastructure (trueno, trueno-db, aprender).
//!
//! # Quick Start
//!
//! ```no_run
//! use trueno_graph::{CsrGraph, NodeId};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Build graph from edge list
//! let mut graph = CsrGraph::new();
//! graph.add_edge(NodeId(0), NodeId(1), 1.0)?;  // main → parse_args
//! graph.add_edge(NodeId(0), NodeId(2), 1.0)?;  // main → validate
//!
//! // Query neighbors (O(1) via CSR indexing)
//! let callees = graph.outgoing_neighbors(NodeId(0))?;
//! assert_eq!(callees.len(), 2);
//!
//! // Save to Parquet (requires "storage" feature, enabled by default)
//! #[cfg(feature = "storage")]
//! graph.write_parquet("graph.parquet").await?;
//!
//! // Load from disk
//! #[cfg(feature = "storage")]
//! let loaded = CsrGraph::read_parquet("graph.parquet").await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Architecture
//!
//! - **Storage**: CSR (Compressed Sparse Row) format for graphs
//! - **Persistence**: Parquet-backed (via trueno-db patterns)
//! - **Algorithms**: Delegates to aprender (`PageRank`, Louvain, centrality)
//! - **Performance**: 25-250x speedups vs CPU baselines (GPU mode)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod algorithms;
pub mod storage;

// GPU acceleration (Phase 3 - optional)
#[cfg(feature = "gpu")]
pub mod gpu;

// Re-export core types
pub use algorithms::{
    bfs, connected_components, dijkstra, dijkstra_path, find_callers, find_patterns, is_cyclic,
    kosaraju_scc, louvain, pagerank, toposort, CommunityDetectionResult, Pattern, PatternMatch,
    Severity,
};
pub use storage::{CsrGraph, NodeId};

#[cfg(feature = "gpu")]
pub use gpu::{
    gpu_bfs, gpu_bfs_paged, GpuBfsResult, GpuCsrBuffers, GpuDevice, GpuMemoryLimits, GraphTile,
    LruTileCache, PagingCoordinator, DEFAULT_MORSEL_SIZE,
};

// Error type
pub use anyhow::{Error, Result};
