//! Quantization parameters for int8 scalar quantization
//!
//! Implements symmetric quantization: Q(x) = round(x / scale)
//! Following Jacob et al. (2018) and Wu et al. (2020)

use super::error::QuantizationError;

/// Quantization parameters for int8 scalar quantization
///
/// Implements symmetric quantization: Q(x) = round(x / scale)
/// Following Jacob et al. (2018) and Wu et al. (2020)
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationParams {
    /// Scale factor: absmax / 127.0 for symmetric quantization
    pub scale: f32,
    /// Zero point (0 for symmetric quantization)
    pub zero_point: i8,
    /// Original embedding dimensions
    pub dims: usize,
}

impl QuantizationParams {
    /// Create new quantization parameters
    ///
    /// # Errors
    /// Returns error if scale is invalid (zero, negative, or non-finite)
    pub fn new(scale: f32, dims: usize) -> Result<Self, QuantizationError> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(QuantizationError::InvalidScale { scale });
        }
        Ok(Self {
            scale,
            zero_point: 0, // Symmetric quantization
            dims,
        })
    }

    /// Create from absmax value (symmetric quantization)
    pub fn from_absmax(absmax: f32, dims: usize) -> Result<Self, QuantizationError> {
        let scale = absmax / 127.0;
        Self::new(scale, dims)
    }

    /// Quantize a single f32 value to i8
    #[inline]
    pub fn quantize_value(&self, value: f32) -> i8 {
        let q = (value / self.scale).round() as i32;
        q.clamp(-128, 127) as i8
    }

    /// Dequantize a single i8 value to f32
    #[inline]
    pub fn dequantize_value(&self, value: i8) -> f32 {
        (value as f32 - self.zero_point as f32) * self.scale
    }

    /// Maximum quantization error bound: scale / 2
    pub fn max_error_bound(&self) -> f32 {
        self.scale / 2.0
    }
}
