
impl DataCompression {
    /// Create Zstd compression with default level
    #[must_use]
    pub const fn zstd() -> Self {
        Self::Zstd { level: 3 }
    }

    /// Create Zstd compression with custom level
    #[must_use]
    pub const fn zstd_level(level: u8) -> Self {
        Self::Zstd { level }
    }

    /// Create delta+Zstd compression
    #[must_use]
    pub const fn delta_zstd() -> Self {
        Self::DeltaZstd { level: 3 }
    }

    /// Create quantized entropy compression
    #[must_use]
    pub const fn quantized(bits: u8) -> Self {
        Self::QuantizedEntropy { bits }
    }

    /// Create sparse compression with threshold
    #[must_use]
    pub fn sparse(threshold: f32) -> Self {
        Self::Sparse {
            threshold: threshold.to_bits(),
        }
    }

    /// Human-readable name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Zstd { .. } => "zstd",
            Self::DeltaZstd { .. } => "delta-zstd",
            Self::QuantizedEntropy { .. } => "quantized-entropy",
            Self::Sparse { .. } => "sparse",
        }
    }

    /// Estimated compression ratio (typical)
    #[must_use]
    pub const fn estimated_ratio(&self) -> f32 {
        match self {
            Self::None => 1.0,
            Self::Zstd { level } => {
                // Higher levels = better compression
                if *level < 5 {
                    2.5
                } else if *level < 10 {
                    4.0
                } else {
                    6.0
                }
            }
            Self::DeltaZstd { level } => {
                if *level < 5 {
                    8.0
                } else {
                    12.0
                }
            }
            Self::QuantizedEntropy { bits } => match bits {
                4 => 8.0,
                8 => 4.0,
                _ => 2.0,
            },
            Self::Sparse { .. } => 5.0, // depends on actual sparsity
        }
    }
}

/// Errors during data embedding operations
#[derive(Debug, Clone)]
pub enum EmbedError {
    /// Data shape doesn't match declared dimensions
    ShapeMismatch { expected: usize, actual: usize },
    /// Target vector length doesn't match samples
    TargetMismatch { expected: usize, actual: usize },
    /// Invalid value (NaN or Inf)
    InvalidValue { index: usize, value: f32 },
    /// Compression error
    CompressionFailed {
        strategy: &'static str,
        message: String,
    },
    /// Decompression error
    DecompressionFailed {
        strategy: &'static str,
        message: String,
    },
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "Shape mismatch: expected {expected} elements, got {actual}"
                )
            }
            Self::TargetMismatch { expected, actual } => {
                write!(
                    f,
                    "Target mismatch: expected {expected} samples, got {actual}"
                )
            }
            Self::InvalidValue { index, value } => {
                write!(f, "Invalid value at index {index}: {value}")
            }
            Self::CompressionFailed { strategy, message } => {
                write!(f, "Compression ({strategy}) failed: {message}")
            }
            Self::DecompressionFailed { strategy, message } => {
                write!(f, "Decompression ({strategy}) failed: {message}")
            }
        }
    }
}

impl std::error::Error for EmbedError {}

/// Simple timestamp without chrono dependency
fn chrono_lite_timestamp() -> String {
    // Returns a placeholder; in real code would use actual timestamp
    "2025-01-01T00:00:00Z".to_string()
}
