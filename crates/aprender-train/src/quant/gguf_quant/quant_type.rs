//! GGUF quantization type enum

/// Quantization type enum for GGUF export
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GGUFQuantType {
    /// 4-bit quantization
    Q4_0,
    /// 8-bit quantization
    Q8_0,
}

impl GGUFQuantType {
    /// Get bytes per block for this quantization type
    pub fn bytes_per_block(&self) -> usize {
        match self {
            GGUFQuantType::Q4_0 => 18, // 2 (scale) + 16 (data)
            GGUFQuantType::Q8_0 => 34, // 2 (scale) + 32 (data)
        }
    }

    /// Get bits per value
    pub fn bits(&self) -> usize {
        match self {
            GGUFQuantType::Q4_0 => 4,
            GGUFQuantType::Q8_0 => 8,
        }
    }

    /// Get theoretical compression ratio vs f32
    pub fn theoretical_compression(&self) -> f32 {
        32.0 / self.bits() as f32
    }
}
