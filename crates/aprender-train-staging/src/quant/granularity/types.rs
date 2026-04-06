//! Quantization granularity and mode type definitions

use serde::{Deserialize, Serialize};

/// Quantization granularity options
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QuantGranularity {
    /// Single scale/zero-point for entire tensor
    #[default]
    PerTensor,
    /// Separate scale/zero-point per channel (axis 0 for weights)
    PerChannel,
    /// Separate scale/zero-point per group of n elements
    PerGroup(usize),
}

/// Quantization mode: symmetric or asymmetric
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QuantMode {
    /// Symmetric: zero-point = 0, range = [-max_abs, max_abs]
    #[default]
    Symmetric,
    /// Asymmetric: zero-point != 0, range = [min, max]
    Asymmetric,
}
