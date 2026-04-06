//! Pruning method enumeration.

use serde::{Deserialize, Serialize};

/// Pruning method selection.
///
/// Each method has different trade-offs between accuracy and computational cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PruneMethod {
    /// Magnitude-based pruning (Han et al., 2015)
    /// Simple, fast, no calibration required.
    #[default]
    Magnitude,

    /// Wanda: Weight and Activation pruning (Sun et al., 2023)
    /// Requires calibration data for activation statistics.
    Wanda,

    /// SparseGPT: Hessian-based pruning (Frantar & Alistarh, 2023)
    /// Most accurate but computationally expensive.
    SparseGpt,

    /// Minitron depth pruning - removes entire layers.
    MinitronDepth,

    /// Minitron width pruning - removes channels.
    MinitronWidth,
}

impl PruneMethod {
    /// Check if this method requires calibration data.
    pub fn requires_calibration(&self) -> bool {
        matches!(
            self,
            PruneMethod::Wanda
                | PruneMethod::SparseGpt
                | PruneMethod::MinitronDepth
                | PruneMethod::MinitronWidth
        )
    }

    /// Get the display name for this method.
    pub fn display_name(&self) -> &'static str {
        match self {
            PruneMethod::Magnitude => "Magnitude",
            PruneMethod::Wanda => "Wanda",
            PruneMethod::SparseGpt => "SparseGPT",
            PruneMethod::MinitronDepth => "Minitron (Depth)",
            PruneMethod::MinitronWidth => "Minitron (Width)",
        }
    }
}
