//! Quantization parameters and quantized tensor structures

use serde::{Deserialize, Serialize};

use super::{QuantGranularity, QuantMode};

/// Quantization parameters for a tensor
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantParams {
    /// Scale factor(s)
    pub scales: Vec<f32>,
    /// Zero point(s) - empty for symmetric quantization
    pub zero_points: Vec<i32>,
    /// Quantization granularity
    pub granularity: QuantGranularity,
    /// Quantization mode
    pub mode: QuantMode,
    /// Bit width (4 or 8)
    pub bits: u8,
}

impl QuantParams {
    /// Get number of scale/zero-point groups
    pub fn num_groups(&self) -> usize {
        self.scales.len()
    }

    /// Check if asymmetric quantization
    pub fn is_asymmetric(&self) -> bool {
        self.mode == QuantMode::Asymmetric
    }
}

/// Quantized tensor with per-channel or per-tensor quantization
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizedTensor {
    /// Quantized integer data (8-bit representation)
    pub data: Vec<i8>,
    /// Quantization parameters
    pub params: QuantParams,
    /// Original shape
    pub shape: Vec<usize>,
}

impl QuantizedTensor {
    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let data_bytes = self.data.len();
        let scale_bytes = self.params.scales.len() * 4;
        let zp_bytes = self.params.zero_points.len() * 4;
        data_bytes + scale_bytes + zp_bytes
    }
}
