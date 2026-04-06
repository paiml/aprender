#![allow(missing_docs)]
//! Execution Path Graph Types (PAR-201)
//!
//! Node, edge, and transfer types for the execution hierarchy.

use super::BrickId;

/// Node ID in the execution graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionNodeId(pub u32);

impl ExecutionNodeId {
    /// Maximum node ID budget (100k nodes).
    pub const MAX_BUDGET: u32 = 100_000;

    /// Validate this node ID is within budget.
    #[inline]
    pub fn validate(self) -> bool {
        debug_assert!(
            self.0 < Self::MAX_BUDGET,
            "CB-BUDGET: node id {} exceeds max budget {}",
            self.0,
            Self::MAX_BUDGET
        );
        self.0 < Self::MAX_BUDGET
    }
}

/// Execution graph node types.
///
/// PAR-201: Represents different levels of the execution hierarchy.
#[derive(Debug, Clone)]
pub enum ExecutionNode {
    /// High-level brick (BrickId from v2)
    Brick { id: BrickId, timing_ns: u64, elements: u64 },
    /// GPU kernel launch
    Kernel {
        name: String,
        /// FNV-1a hash of PTX source for identity
        ptx_hash: u64,
        /// Grid dimensions (blocks)
        grid: (u32, u32, u32),
        /// Block dimensions (threads)
        block: (u32, u32, u32),
        /// Shared memory bytes
        shared_mem: u32,
        /// Kernel execution time in nanoseconds (Phase 9: for CPA)
        timing_ns: Option<u64>,
        /// Arithmetic intensity (FLOPs/byte) for roofline analysis (Phase 9)
        arithmetic_intensity: Option<f32>,
        /// Achieved throughput in TFLOP/s (Phase 9)
        achieved_tflops: Option<f32>,
    },
    /// Memory transfer operation (Phase 9: data movement topology)
    Transfer {
        /// Source location description
        src: String,
        /// Destination location description
        dst: String,
        /// Bytes transferred
        bytes: u64,
        /// Transfer direction
        direction: TransferDirection,
        /// Transfer time in nanoseconds
        timing_ns: Option<u64>,
    },
    /// Rust function (from DWARF or manual annotation)
    Function { name: String, file: Option<String>, line: Option<u32> },
    /// Transformer layer grouping
    Layer { index: u32 },
    /// Phase 11 (E.9.4): Async task metrics for poll efficiency tracking
    AsyncTask {
        /// Task name for identification
        name: String,
        /// Number of times poll() was called
        poll_count: u64,
        /// Number of times poll() returned Pending
        yield_count: u64,
        /// Total time spent in poll() (nanoseconds)
        total_poll_ns: u64,
    },
}

impl ExecutionNode {
    /// Get the display name of this node.
    pub fn name(&self) -> String {
        match self {
            Self::Brick { id, .. } => id.name().to_string(),
            Self::Kernel { name, .. } => name.clone(),
            Self::Function { name, .. } => name.clone(),
            Self::Layer { index } => format!("Layer{}", index),
            Self::Transfer { src, dst, direction, .. } => {
                let dir = match direction {
                    TransferDirection::H2D => "H2D",
                    TransferDirection::D2H => "D2H",
                    TransferDirection::D2D => "D2D",
                };
                format!("{}:{}->{}", dir, src, dst)
            }
            Self::AsyncTask { name, .. } => name.clone(),
        }
    }

    /// Check if this is a kernel node.
    pub fn is_kernel(&self) -> bool {
        matches!(self, Self::Kernel { .. })
    }

    /// Check if this is a brick node.
    pub fn is_brick(&self) -> bool {
        matches!(self, Self::Brick { .. })
    }

    /// Check if this is a transfer node.
    pub fn is_transfer(&self) -> bool {
        matches!(self, Self::Transfer { .. })
    }

    /// Get timing if available (bricks, kernels, and transfers).
    pub fn timing_ns(&self) -> Option<u64> {
        match self {
            Self::Brick { timing_ns, .. } => Some(*timing_ns),
            Self::Kernel { timing_ns, .. } => *timing_ns,
            Self::Transfer { timing_ns, .. } => *timing_ns,
            _ => None,
        }
    }

    /// Get PTX hash if available (kernels only).
    pub fn ptx_hash(&self) -> Option<u64> {
        match self {
            Self::Kernel { ptx_hash, .. } => Some(*ptx_hash),
            _ => None,
        }
    }

    /// Get arithmetic intensity if available (kernels only, Phase 9).
    pub fn arithmetic_intensity(&self) -> Option<f32> {
        match self {
            Self::Kernel { arithmetic_intensity, .. } => *arithmetic_intensity,
            _ => None,
        }
    }

    /// Get achieved TFLOP/s if available (kernels only, Phase 9).
    pub fn achieved_tflops(&self) -> Option<f32> {
        match self {
            Self::Kernel { achieved_tflops, .. } => *achieved_tflops,
            _ => None,
        }
    }

    /// Get transfer bytes if available (transfers only, Phase 9).
    pub fn transfer_bytes(&self) -> Option<u64> {
        match self {
            Self::Transfer { bytes, .. } => Some(*bytes),
            _ => None,
        }
    }
}

/// Edge types in execution graph.
///
/// PAR-201: Describes relationships between execution nodes.
/// Phase 9 (E.7.12): Added DependsOn and Transfer for advanced profiling.
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeType {
    /// Function calls function
    Calls,
    /// Brick contains sub-operations
    Contains,
    /// Function launches GPU kernel
    Launches,
    /// Temporal sequence (A happens before B)
    Sequence,
    /// Dependency edge for critical path analysis (CUDA events, stream sync)
    /// PAR-201 Phase 9: CPA requires tracking true dependencies vs containment
    DependsOn,
    /// Data transfer edge with byte count (H2D/D2H/D2D)
    /// PAR-201 Phase 9: For data movement topology and ping-pong detection
    Transfer {
        /// Bytes transferred
        bytes: u64,
        /// Transfer direction
        direction: TransferDirection,
    },
}

/// Direction of memory transfer.
///
/// PAR-201 Phase 9: Used with EdgeType::Transfer for data movement analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// Host to Device
    H2D,
    /// Device to Host
    D2H,
    /// Device to Device
    D2D,
}

/// An edge in the execution graph.
#[derive(Debug, Clone)]
pub struct ExecutionEdge {
    /// Source node ID
    pub src: ExecutionNodeId,
    /// Destination node ID
    pub dst: ExecutionNodeId,
    /// Edge type
    pub edge_type: EdgeType,
    /// Optional weight (e.g., call count, timing)
    pub weight: f32,
}
