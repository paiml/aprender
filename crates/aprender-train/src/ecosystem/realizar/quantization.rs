//! Quantization types for GGUF export.

use serde::{Deserialize, Serialize};

/// Quantization type for GGUF export.
///
/// These correspond to llama.cpp quantization types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationType {
    /// 4-bit quantization with k-quants (recommended for most use cases)
    Q4KM,
    /// 5-bit quantization with k-quants (higher quality than Q4_K_M)
    Q5KM,
    /// 8-bit quantization (highest quality, larger size)
    Q80,
    /// 16-bit floating point (no quantization)
    F16,
    /// 32-bit floating point (no quantization, largest)
    F32,
    /// 2-bit quantization (extreme compression, quality loss)
    Q2K,
    /// 3-bit quantization (aggressive compression)
    Q3KM,
    /// 6-bit quantization (high quality)
    Q6K,
}

impl QuantizationType {
    /// Get the GGUF type string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Q4KM => "Q4_K_M",
            Self::Q5KM => "Q5_K_M",
            Self::Q80 => "Q8_0",
            Self::F16 => "F16",
            Self::F32 => "F32",
            Self::Q2K => "Q2_K",
            Self::Q3KM => "Q3_K_M",
            Self::Q6K => "Q6_K",
        }
    }

    /// Get estimated bits per weight.
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            Self::Q2K => 2.5,
            Self::Q3KM => 3.5,
            Self::Q4KM => 4.5,
            Self::Q5KM => 5.5,
            Self::Q6K => 6.5,
            Self::Q80 => 8.0,
            Self::F16 => 16.0,
            Self::F32 => 32.0,
        }
    }

    /// Get relative quality score (0-100).
    pub fn quality_score(&self) -> u8 {
        match self {
            Self::Q2K => 50,
            Self::Q3KM => 65,
            Self::Q4KM => 78,
            Self::Q5KM => 85,
            Self::Q6K => 92,
            Self::Q80 => 97,
            Self::F16 => 100,
            Self::F32 => 100,
        }
    }

    /// Estimate output size given input size in bytes.
    pub fn estimate_size(&self, original_bytes: u64) -> u64 {
        let ratio = self.bits_per_weight() / 32.0;
        (original_bytes as f32 * ratio) as u64
    }

    /// Parse from string (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        let normalized = s.to_uppercase().replace(['-', '_'], "");
        match normalized.as_str() {
            "Q4KM" | "Q4K" => Some(Self::Q4KM),
            "Q5KM" | "Q5K" => Some(Self::Q5KM),
            "Q80" | "Q8" => Some(Self::Q80),
            "F16" | "FP16" => Some(Self::F16),
            "F32" | "FP32" => Some(Self::F32),
            "Q2K" | "Q2" => Some(Self::Q2K),
            "Q3KM" | "Q3K" => Some(Self::Q3KM),
            "Q6K" | "Q6" => Some(Self::Q6K),
            _ => None,
        }
    }
}

impl std::fmt::Display for QuantizationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
