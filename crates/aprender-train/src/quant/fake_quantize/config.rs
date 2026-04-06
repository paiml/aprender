//! Fake quantization configuration types.

/// Fake quantization configuration
#[derive(Clone, Debug)]
pub struct FakeQuantConfig {
    /// Number of bits for quantization (e.g., 4, 8)
    pub bits: usize,
    /// Whether quantization is symmetric (centered at 0)
    pub symmetric: bool,
    /// Quantization range: min value
    pub qmin: i32,
    /// Quantization range: max value
    pub qmax: i32,
}

impl FakeQuantConfig {
    /// Create symmetric fake quantization config
    ///
    /// # Arguments
    /// * `bits` - Number of bits (4-bit: qmin=-7, qmax=7; 8-bit: qmin=-127, qmax=127)
    pub fn symmetric(bits: usize) -> Self {
        let qmax = (1 << (bits - 1)) - 1; // 2^(bits-1) - 1
        let qmin = -qmax;
        Self { bits, symmetric: true, qmin, qmax }
    }

    /// Create asymmetric fake quantization config
    ///
    /// # Arguments
    /// * `bits` - Number of bits (4-bit: qmin=0, qmax=15; 8-bit: qmin=0, qmax=255)
    pub fn asymmetric(bits: usize) -> Self {
        let qmax = (1 << bits) - 1; // 2^bits - 1
        Self { bits, symmetric: false, qmin: 0, qmax }
    }

    /// 4-bit symmetric quantization
    pub fn q4_symmetric() -> Self {
        Self::symmetric(4)
    }

    /// 8-bit symmetric quantization
    pub fn q8_symmetric() -> Self {
        Self::symmetric(8)
    }
}

impl Default for FakeQuantConfig {
    fn default() -> Self {
        Self::q8_symmetric()
    }
}
