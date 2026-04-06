//! Internal tensor metadata types for export.

use serde::{Deserialize, Serialize};

/// Tensor metadata for export
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub(super) struct TensorMetadata {
    /// Tensor name
    pub name: String,
    /// Data type
    pub dtype: DataType,
    /// Shape
    pub shape: Vec<usize>,
    /// Byte offset in file
    pub offset: usize,
    /// Byte size
    pub size: usize,
}

/// Supported data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)]
pub(super) enum DataType {
    F32,
    F16,
    BF16,
    I32,
    I8,
    U8,
    Q4_0,
    Q8_0,
}

#[allow(dead_code)]
impl DataType {
    /// Bytes per element
    #[must_use]
    pub(super) fn bytes_per_element(&self) -> f32 {
        match self {
            Self::F32 | Self::I32 => 4.0,
            Self::F16 | Self::BF16 => 2.0,
            Self::I8 | Self::U8 => 1.0,
            Self::Q4_0 => 0.5,
            Self::Q8_0 => 1.0,
        }
    }
}
