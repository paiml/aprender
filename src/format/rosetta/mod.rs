//! Rosetta Stone - Universal Model Format Converter
//!
//! Named after the ancient artifact that enabled translation between scripts,
//! this module provides bidirectional conversion between ML model formats.
//!
//! # Supported Formats
//!
//! | Format | Extensions | Quantization |
//! |--------|------------|--------------|
//! | GGUF | `.gguf` | Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16, F32 |
//! | SafeTensors | `.safetensors` | F16, F32, BF16 |
//! | APR | `.apr` | Q4_0, Q8_0, F16, F32 |
//!
//! # Conversion Matrix (6 Direct Paths)
//!
//! ```text
//!     GGUF ←──────→ APR ←──────→ SafeTensors
//!       ↑                              ↑
//!       └──────────────────────────────┘
//! ```
//!
//! # Toyota Way Alignment
//!
//! - **Genchi Genbutsu**: Inspect raw tensor data, not abstractions
//! - **Jidoka**: Stop on any conversion anomaly
//! - **Kaizen**: Multi-step chains for iterative improvement
//! - **Visualization**: Full metadata before/after display
//!
//! # Example
//!
//! ```rust,ignore
//! use aprender::format::rosetta::{RosettaStone, FormatType, ConversionOptions};
//!
//! let rosetta = RosettaStone::new();
//!
//! // Inspect before conversion
//! let inspection = rosetta.inspect("model.gguf")?;
//! println!("{}", inspection);
//!
//! // Convert GGUF to APR
//! let report = rosetta.convert(
//!     "model.gguf",
//!     "model.apr",
//!     ConversionOptions::default()
//! )?;
//!
//! // Verify round-trip
//! let verification = rosetta.verify_roundtrip("model.gguf", FormatType::Apr)?;
//! assert!(verification.is_equivalent);
//! ```
//!
//! # References
//!
//! - Popper, K. (1959). The Logic of Scientific Discovery. Routledge.
//! - Ohno, T. (1988). Toyota Production System. Productivity Press.
//! - Gerganov, G. (2023). GGUF Format Specification. llama.cpp.

use crate::error::{AprenderError, Result};
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

/// GH-187: Bug classification for common format conversion failures.
///
/// Used in validation and differential tracing to quickly identify the
/// category of failure without manual tensor inspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BugClassification {
    /// Embedding tensor stored as [hidden, vocab] instead of [vocab, hidden]
    EmbeddingTransposed,
    /// Weight tensor has all zeros — packing bug or uninitialized memory
    WeightAllZeros,
    /// Tensor shape doesn't match expected dimensions for the architecture
    ShapeMismatch,
    /// NaN/Inf values in tensor data — numerical instability
    NumericalCorruption,
    /// Tensor dtype doesn't match what the loader expects
    DtypeMismatch,
}

impl fmt::Display for BugClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmbeddingTransposed => write!(f, "EMBEDDING_TRANSPOSED"),
            Self::WeightAllZeros => write!(f, "WEIGHT_ALL_ZEROS"),
            Self::ShapeMismatch => write!(f, "SHAPE_MISMATCH"),
            Self::NumericalCorruption => write!(f, "NUMERICAL_CORRUPTION"),
            Self::DtypeMismatch => write!(f, "DTYPE_MISMATCH"),
        }
    }
}

/// Bug 212: Check if a path is a sharded SafeTensors index file.
fn is_sharded_index(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.ends_with(".index.json"))
}

// ============================================================================
// Format Types
// ============================================================================

/// Supported model formats for Rosetta Stone conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatType {
    /// GGUF - llama.cpp compatible format
    Gguf,
    /// SafeTensors - HuggingFace format
    SafeTensors,
    /// APR - Aprender native format
    Apr,
}


include!("mod_include_01.rs");
