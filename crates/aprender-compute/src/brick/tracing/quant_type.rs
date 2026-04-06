// ============================================================================
// QuantType - Quantization type tracking
// ============================================================================

/// Quantization type for tracking quantization errors (MLT-04).
///
/// Note: Variant names follow GGML conventions (e.g., Q4_K) for interoperability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[allow(non_camel_case_types)]
pub enum QuantType {
    /// Full precision (FP32)
    #[default]
    F32,
    /// Half precision (FP16)
    F16,
    /// Brain floating point (BF16)
    Bf16,
    /// 8-bit integer quantization
    Q8_0,
    /// 4-bit quantization (GGML)
    Q4_0,
    /// 4-bit quantization with k-quants
    Q4_K,
    /// 5-bit quantization with k-quants
    Q5_K,
    /// 6-bit quantization with k-quants
    Q6_K,
    /// 2-bit quantization
    Q2_K,
    /// 3-bit quantization
    Q3_K,
}

impl QuantType {
    /// Get bits per element for this quantization type.
    pub fn bits_per_element(self) -> f32 {
        match self {
            Self::F32 => 32.0,
            Self::F16 | Self::Bf16 => 16.0,
            Self::Q8_0 => 8.0,
            Self::Q6_K => 6.5,
            Self::Q5_K => 5.5,
            Self::Q4_0 | Self::Q4_K => 4.5,
            Self::Q3_K => 3.5,
            Self::Q2_K => 2.5,
        }
    }

    /// Get compression ratio vs FP32.
    pub fn compression_ratio(self) -> f32 {
        32.0 / self.bits_per_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_type_bits() {
        assert_eq!(QuantType::F32.bits_per_element(), 32.0);
        assert_eq!(QuantType::F16.bits_per_element(), 16.0);
        assert_eq!(QuantType::Q8_0.bits_per_element(), 8.0);
        assert_eq!(QuantType::Q4_K.bits_per_element(), 4.5);
    }

    #[test]
    fn test_quant_type_compression_ratio() {
        // F32 -> F32 = 1x
        assert!((QuantType::F32.compression_ratio() - 1.0).abs() < 0.01);
        // F32 -> F16 = 2x
        assert!((QuantType::F16.compression_ratio() - 2.0).abs() < 0.01);
        // F32 -> Q4_K = ~7.1x
        assert!(QuantType::Q4_K.compression_ratio() > 7.0);
    }
}
