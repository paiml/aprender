//! Precision type definitions for mixed-precision training.

use std::fmt;

/// Data type precision levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Precision {
    /// 32-bit floating point (default)
    #[default]
    Fp32,
    /// 16-bit floating point (IEEE half precision)
    Fp16,
    /// 16-bit brain floating point (truncated mantissa)
    Bf16,
}

impl Precision {
    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            Precision::Fp32 => 4,
            Precision::Fp16 | Precision::Bf16 => 2,
        }
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Precision::Fp32 => "fp32",
            Precision::Fp16 => "fp16",
            Precision::Bf16 => "bf16",
        }
    }

    /// Whether this is a reduced precision type
    pub fn is_reduced(&self) -> bool {
        matches!(self, Precision::Fp16 | Precision::Bf16)
    }

    /// Memory multiplier compared to fp32
    pub fn memory_multiplier(&self) -> f32 {
        match self {
            Precision::Fp32 => 1.0,
            Precision::Fp16 | Precision::Bf16 => 0.5,
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}
