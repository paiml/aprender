//! Sparsity pattern configuration.

use serde::{Deserialize, Serialize};

/// Sparsity pattern selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SparsityPatternConfig {
    /// Unstructured sparsity - any weight can be pruned.
    #[default]
    Unstructured,

    /// N:M structured sparsity (e.g., 2:4 for NVIDIA Ampere).
    #[serde(rename = "nm")]
    NM {
        /// Number of non-zero elements per group.
        n: usize,
        /// Group size.
        m: usize,
    },

    /// Block sparsity - entire blocks pruned together.
    Block {
        /// Block height.
        height: usize,
        /// Block width.
        width: usize,
    },

    /// Row sparsity - entire output channels pruned.
    Row,

    /// Column sparsity - entire input channels pruned.
    Column,
}

impl SparsityPatternConfig {
    /// Create 2:4 sparsity pattern for NVIDIA Ampere.
    pub fn nm_2_4() -> Self {
        SparsityPatternConfig::NM { n: 2, m: 4 }
    }

    /// Create 4:8 sparsity pattern.
    pub fn nm_4_8() -> Self {
        SparsityPatternConfig::NM { n: 4, m: 8 }
    }

    /// Get the theoretical sparsity for this pattern.
    pub fn theoretical_sparsity(&self) -> f32 {
        match self {
            SparsityPatternConfig::Unstructured => 0.0, // Variable
            SparsityPatternConfig::NM { n, m } => 1.0 - (*n as f32 / *m as f32),
            SparsityPatternConfig::Block { .. } => 0.0, // Variable
            SparsityPatternConfig::Row => 0.0,          // Variable
            SparsityPatternConfig::Column => 0.0,       // Variable
        }
    }
}
