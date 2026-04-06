//! Execution Graph Node Types and Profiling Primitives
//!
//! This module contains all type definitions for execution path tracking:
//!
//! - **PAR-073**: BrickSample, BrickBottleneck - foundational profiling primitives
//! - **PAR-200**: BrickId, BrickCategory, SyncMode - O(1) hot path brick identification
//! - **PAR-201**: ExecutionNode, EdgeType, etc. - execution hierarchy types

use std::fmt;

mod execution;
mod stats;

pub use execution::{EdgeType, ExecutionEdge, ExecutionNode, ExecutionNodeId, TransferDirection};
pub use stats::{BrickStats, CategoryStats, PtxRegistry};

// ============================================================================
// BrickProfiler: FOUNDATIONAL Real-Time Per-Brick Timing (PAR-073)
// ============================================================================

/// Individual brick timing sample.
/// Pure Rust timing using `std::time::Instant`.
#[derive(Debug, Clone, Copy)]
pub struct BrickSample {
    /// Brick name hash (for fast lookup)
    pub brick_id: u64,
    /// Elapsed time in nanoseconds
    pub elapsed_ns: u64,
    /// Number of elements processed
    pub elements: u64,
}

/// Bottleneck classification for roofline analysis (PMAT-451)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BrickBottleneck {
    /// Not classified
    #[default]
    Unknown,
    /// Limited by memory bandwidth
    Memory,
    /// Limited by compute throughput
    Compute,
}

impl fmt::Display for BrickBottleneck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BrickBottleneck::Unknown => write!(f, "unknown"),
            BrickBottleneck::Memory => write!(f, "memory"),
            BrickBottleneck::Compute => write!(f, "compute"),
        }
    }
}

// ============================================================================
// PAR-200: BrickProfiler v2 - O(1) Hot Path with BrickId Enum
// ============================================================================

/// Well-known brick types for O(1) lookup on hot path.
///
/// PAR-200: Eliminates string allocation and HashMap hashing during profiling.
/// Use `BrickId::Custom` with string fallback for unknown brick types.
///
/// # Example
/// ```rust
/// use trueno::brick::BrickId;
///
/// let brick = BrickId::RmsNorm;
/// assert_eq!(brick.category(), trueno::brick::BrickCategory::Norm);
/// assert_eq!(brick.name(), "RmsNorm");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BrickId {
    // Normalization (0-1)
    /// RMS normalization layer
    RmsNorm = 0,
    /// Layer normalization
    LayerNorm = 1,

    // Attention (2-7)
    /// Q/K/V projection (combined or separate)
    QkvProjection = 2,
    /// Rotary position embedding
    RopeEmbedding = 3,
    /// Attention score computation (Q @ K^T)
    AttentionScore = 4,
    /// Attention softmax
    AttentionSoftmax = 5,
    /// Attention output (scores @ V)
    AttentionOutput = 6,
    /// Output projection after attention
    OutputProjection = 7,

    // FFN (8-11)
    /// Gate projection (for gated FFN)
    GateProjection = 8,
    /// Up projection
    UpProjection = 9,
    /// SiLU/GELU/ReLU activation
    Activation = 10,
    /// Down projection
    DownProjection = 11,

    // Other (12-14)
    /// Token embedding lookup
    Embedding = 12,
    /// Language model head (logits)
    LmHead = 13,
    /// Token sampling
    Sampling = 14,

    // Sparse (15-17) — CUDA-parity-spec Phase 1
    /// Sparse matrix-vector multiply (cuSPARSE parity)
    SpMV = 15,
    /// Sparse matrix-dense matrix multiply
    SpMM = 16,
    /// Sparse format conversion
    FormatConvert = 17,

    // FFT (18-19) — CUDA-parity-spec Phase 2
    /// 1D Fast Fourier Transform
    FFT1D = 18,
    /// 2D Fast Fourier Transform
    FFT2D = 19,

    // Solvers (20-22) — CUDA-parity-spec Phase 2
    /// LU factorization
    LUFactorize = 20,
    /// QR factorization
    QRFactorize = 21,
    /// Singular Value Decomposition
    SVDCompute = 22,
}

impl BrickId {
    /// Number of well-known brick types.
    pub const COUNT: usize = 23;

    /// All BrickId variants in order, for safe index-based iteration.
    ///
    /// Eliminates need for `transmute::<u8, BrickId>` in array initialization.
    pub const ALL: [BrickId; Self::COUNT] = [
        Self::RmsNorm,
        Self::LayerNorm,
        Self::QkvProjection,
        Self::RopeEmbedding,
        Self::AttentionScore,
        Self::AttentionSoftmax,
        Self::AttentionOutput,
        Self::OutputProjection,
        Self::GateProjection,
        Self::UpProjection,
        Self::Activation,
        Self::DownProjection,
        Self::Embedding,
        Self::LmHead,
        Self::Sampling,
        Self::SpMV,
        Self::SpMM,
        Self::FormatConvert,
        Self::FFT1D,
        Self::FFT2D,
        Self::LUFactorize,
        Self::QRFactorize,
        Self::SVDCompute,
    ];

    /// Validate that a raw u8 is within the BrickId range.
    #[inline]
    pub fn validate_index(index: usize) -> bool {
        debug_assert!(
            index < Self::COUNT,
            "CB-BUDGET: brick index {} out of bounds (max {})",
            index,
            Self::COUNT
        );
        index < Self::COUNT
    }

    /// Get the category for hierarchical aggregation.
    #[inline]
    pub fn category(self) -> BrickCategory {
        match self {
            Self::RmsNorm | Self::LayerNorm => BrickCategory::Norm,
            Self::QkvProjection
            | Self::RopeEmbedding
            | Self::AttentionScore
            | Self::AttentionSoftmax
            | Self::AttentionOutput
            | Self::OutputProjection => BrickCategory::Attention,
            Self::GateProjection | Self::UpProjection | Self::Activation | Self::DownProjection => {
                BrickCategory::Ffn
            }
            Self::Embedding | Self::LmHead | Self::Sampling => BrickCategory::Other,
            Self::SpMV | Self::SpMM | Self::FormatConvert => BrickCategory::Sparse,
            Self::FFT1D | Self::FFT2D => BrickCategory::Fft,
            Self::LUFactorize | Self::QRFactorize | Self::SVDCompute => BrickCategory::Solver,
        }
    }

    /// Get the string name of this brick.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::RmsNorm => "RmsNorm",
            Self::LayerNorm => "LayerNorm",
            Self::QkvProjection => "QkvProjection",
            Self::RopeEmbedding => "RopeEmbedding",
            Self::AttentionScore => "AttentionScore",
            Self::AttentionSoftmax => "AttentionSoftmax",
            Self::AttentionOutput => "AttentionOutput",
            Self::OutputProjection => "OutputProjection",
            Self::GateProjection => "GateProjection",
            Self::UpProjection => "UpProjection",
            Self::Activation => "Activation",
            Self::DownProjection => "DownProjection",
            Self::Embedding => "Embedding",
            Self::LmHead => "LmHead",
            Self::Sampling => "Sampling",
            Self::SpMV => "SpMV",
            Self::SpMM => "SpMM",
            Self::FormatConvert => "FormatConvert",
            Self::FFT1D => "FFT1D",
            Self::FFT2D => "FFT2D",
            Self::LUFactorize => "LUFactorize",
            Self::QRFactorize => "QRFactorize",
            Self::SVDCompute => "SVDCompute",
        }
    }

    /// Try to parse a string into a BrickId.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "RmsNorm" => Some(Self::RmsNorm),
            "LayerNorm" => Some(Self::LayerNorm),
            "QkvProjection" | "Qkv" => Some(Self::QkvProjection),
            "RopeEmbedding" | "Rope" | "RoPE" => Some(Self::RopeEmbedding),
            "AttentionScore" => Some(Self::AttentionScore),
            "AttentionSoftmax" | "Softmax" => Some(Self::AttentionSoftmax),
            "AttentionOutput" => Some(Self::AttentionOutput),
            "OutputProjection" | "OutProj" => Some(Self::OutputProjection),
            "GateProjection" | "Gate" => Some(Self::GateProjection),
            "UpProjection" | "Up" => Some(Self::UpProjection),
            "Activation" | "SiLU" | "GELU" | "ReLU" => Some(Self::Activation),
            "DownProjection" | "Down" => Some(Self::DownProjection),
            "Embedding" | "Embed" => Some(Self::Embedding),
            "LmHead" | "Head" => Some(Self::LmHead),
            "Sampling" | "Sample" => Some(Self::Sampling),
            "SpMV" | "spmv" => Some(Self::SpMV),
            "SpMM" | "spmm" => Some(Self::SpMM),
            "FormatConvert" => Some(Self::FormatConvert),
            "FFT1D" | "fft1d" | "FFT" => Some(Self::FFT1D),
            "FFT2D" | "fft2d" => Some(Self::FFT2D),
            "LUFactorize" | "LU" => Some(Self::LUFactorize),
            "QRFactorize" | "QR" => Some(Self::QRFactorize),
            "SVDCompute" | "SVD" => Some(Self::SVDCompute),
            _ => None,
        }
    }
}

impl fmt::Display for BrickId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Category for hierarchical aggregation of brick statistics.
///
/// PAR-200: Groups related bricks for high-level performance analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum BrickCategory {
    /// Normalization layers (RmsNorm, LayerNorm)
    Norm = 0,
    /// Attention mechanism (QKV, RoPE, scores, softmax, output)
    Attention = 1,
    /// Feed-forward network (gate, up, activation, down)
    Ffn = 2,
    /// Other operations (embedding, lm_head, sampling)
    #[default]
    Other = 3,
    /// Sparse linear algebra (SpMV, SpMM, format conversion)
    Sparse = 4,
    /// FFT operations (1D, 2D, 3D)
    Fft = 5,
    /// Dense solvers (LU, QR, SVD, Cholesky)
    Solver = 6,
}

impl BrickCategory {
    /// Number of categories.
    pub const COUNT: usize = 7;

    /// All BrickCategory variants in order, for safe index-based iteration.
    pub const ALL: [BrickCategory; Self::COUNT] = [
        Self::Norm,
        Self::Attention,
        Self::Ffn,
        Self::Other,
        Self::Sparse,
        Self::Fft,
        Self::Solver,
    ];

    /// Get the string name of this category.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Norm => "Norm",
            Self::Attention => "Attention",
            Self::Ffn => "FFN",
            Self::Other => "Other",
            Self::Sparse => "Sparse",
            Self::Fft => "FFT",
            Self::Solver => "Solver",
        }
    }
}

impl fmt::Display for BrickCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Synchronization mode for GPU profiling.
///
/// PAR-200: Controls the trade-off between accuracy and overhead.
///
/// # Performance Characteristics
///
/// | Mode | Overhead | Accuracy | Use Case |
/// |------|----------|----------|----------|
/// | `Immediate` | ~200% | Exact per-kernel | Debugging |
/// | `PerLayer` | ~20% | Per-layer exact | Development |
/// | `Deferred` | ~5% | Approximate | Production |
/// | `None` | 0% | N/A | Disabled |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncMode {
    /// Sync after each kernel (accurate but slow).
    /// Best for debugging and detailed optimization.
    Immediate,
    /// Sync once per transformer layer.
    /// Good balance for development.
    PerLayer,
    /// Sync once per forward pass (fast, approximate).
    /// Best for production profiling.
    #[default]
    Deferred,
    /// No synchronization (profiling disabled or CPU-only).
    None,
}
