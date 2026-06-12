//! Execution graph methods for BrickProfiler (PAR-201).
//!
//! Extracted from mod.rs to keep file sizes manageable.
//! Contains all `graph_*` methods for execution path tracking.

use super::BrickProfiler;
use crate::brick::exec_graph::{BrickId, ExecutionNode, ExecutionNodeId};

impl BrickProfiler {
    // ========================================================================
    // PAR-201: Execution Path Graph
    // ========================================================================

    /// Enable execution graph tracking.
    ///
    /// When enabled, the profiler records the execution hierarchy:
    /// - Layer → Brick → Kernel relationships
    /// - PTX hashes for kernel identity
    /// - Timing data per node
    pub fn enable_graph(&mut self) {
        self.graph_enabled = true;
    }

    /// Disable execution graph tracking.
    pub fn disable_graph(&mut self) {
        self.graph_enabled = false;
    }

    /// Check if execution graph tracking is enabled.
    #[must_use]
    pub fn is_graph_enabled(&self) -> bool {
        self.graph_enabled
    }

    /// Get the execution graph (immutable).
    #[must_use]
    pub fn execution_graph(&self) -> &crate::brick::exec_graph::ExecutionGraph {
        &self.execution_graph
    }

    /// Get the execution graph (mutable).
    pub fn execution_graph_mut(&mut self) -> &mut crate::brick::exec_graph::ExecutionGraph {
        &mut self.execution_graph
    }

    /// Push a scope for hierarchical graph recording.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// profiler.enable_graph();
    /// profiler.graph_push_scope(ExecutionNode::Layer { index: 0 });
    /// // ... record bricks and kernels ...
    /// profiler.graph_pop_scope();
    /// ```
    pub fn graph_push_scope(&mut self, node: ExecutionNode) -> Option<ExecutionNodeId> {
        if !self.graph_enabled {
            return None;
        }
        debug_assert!(
            self.execution_graph.num_nodes() < 100_000,
            "CB-BUDGET: execution graph has {} nodes, exceeds 100k budget",
            self.execution_graph.num_nodes()
        );
        Some(self.execution_graph.push_scope(node))
    }

    /// Pop the current scope.
    pub fn graph_pop_scope(&mut self) -> Option<ExecutionNodeId> {
        if !self.graph_enabled {
            return None;
        }
        self.execution_graph.pop_scope()
    }

    /// Record a brick in the execution graph.
    ///
    /// This should be called after `stop_brick()` with the timing data.
    pub fn graph_record_brick(
        &mut self,
        brick_id: BrickId,
        timing_ns: u64,
        elements: u64,
    ) -> Option<ExecutionNodeId> {
        if !self.graph_enabled {
            return None;
        }
        let node = ExecutionNode::Brick { id: brick_id, timing_ns, elements };
        Some(self.execution_graph.add_node_in_scope(node))
    }

    /// Record a kernel launch in the execution graph.
    ///
    /// # Arguments
    /// - `name`: Kernel name (e.g., "batched_q4k_gemv")
    /// - `ptx_hash`: FNV-1a hash of PTX source for identity
    /// - `grid`: Grid dimensions (blocks)
    /// - `block`: Block dimensions (threads)
    /// - `shared_mem`: Shared memory bytes
    pub fn graph_record_kernel(
        &mut self,
        name: &str,
        ptx_hash: u64,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
    ) -> Option<ExecutionNodeId> {
        if !self.graph_enabled {
            return None;
        }
        Some(self.execution_graph.record_kernel_launch(name, ptx_hash, grid, block, shared_mem))
    }

    /// Export execution graph to DOT format for visualization.
    ///
    /// Use with Graphviz: `dot -Tsvg output.dot -o graph.svg`
    #[must_use]
    pub fn graph_to_dot(&self) -> String {
        self.execution_graph.to_dot()
    }

    // NOTE: `graph_to_csr()` removed (APR-MONO self-containment) — see
    // exec_graph/traversal/export.rs. CSR export belongs in aprender-graph, not compute.

    /// Clear the execution graph.
    pub fn graph_clear(&mut self) {
        self.execution_graph.clear();
    }

    /// Check if the execution graph scope stack is balanced.
    #[must_use]
    pub fn graph_is_scope_balanced(&self) -> bool {
        self.execution_graph.is_scope_balanced()
    }
}
