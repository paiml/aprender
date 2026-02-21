//! ONNX format reader (GH-238)
//!
//! Lightweight ONNX protobuf parser that extracts tensor initializers
//! (weights) from `.onnx` files without requiring the `prost` crate.
//!
//! # ONNX Protobuf Layout (simplified)
//!
//! ```text
//! ModelProto {
//!   ir_version: int64        (field 1)
//!   graph: GraphProto        (field 7)
//!     initializer: [TensorProto]  (field 5, repeated)
//!       dims: [int64]        (field 1, repeated/packed)
//!       data_type: int32     (field 2)
//!       name: string         (field 8)
//!       raw_data: bytes      (field 13)
//!       float_data: [float]  (field 4, packed)
//! }
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use aprender::format::onnx::OnnxReader;
//!
//! let reader = OnnxReader::from_file("model.onnx")?;
//! for tensor in reader.tensors() {
//!     println!("{}: {:?} ({:?})", tensor.name, tensor.shape, tensor.data_type);
//! }
//! ```

use crate::error::{AprenderError, Result};
use std::collections::BTreeMap;
use std::path::Path;

/// ONNX data types (from onnx.proto3 TensorProto.DataType)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxDataType {
    Float,
    Uint8,
    Int8,
    Uint16,
    Int16,
    Int32,
    Int64,
    String,
    Bool,
    Float16,
    Double,
    Uint32,
    Uint64,
    BFloat16,
    Unknown(i32),
}

impl OnnxDataType {
    fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::Float,
            2 => Self::Uint8,
            3 => Self::Int8,
            4 => Self::Uint16,
            5 => Self::Int16,
            6 => Self::Int32,
            7 => Self::Int64,
            8 => Self::String,
            9 => Self::Bool,
            10 => Self::Float16,
            11 => Self::Double,
            12 => Self::Uint32,
            13 => Self::Uint64,
            16 => Self::BFloat16,
            other => Self::Unknown(other),
        }
    }

    /// Bytes per element for this data type
    pub fn element_size(&self) -> usize {
        match self {
            Self::Float | Self::Int32 | Self::Uint32 => 4,
            Self::Double | Self::Int64 | Self::Uint64 => 8,
            Self::Float16 | Self::BFloat16 | Self::Int16 | Self::Uint16 => 2,
            Self::Uint8 | Self::Int8 | Self::Bool => 1,
            Self::String | Self::Unknown(_) => 0,
        }
    }
}

/// A tensor extracted from an ONNX file
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    /// Tensor name
    pub name: String,
    /// Tensor shape (dimensions)
    pub shape: Vec<usize>,
    /// Data type
    pub data_type: OnnxDataType,
    /// Raw bytes of tensor data
    pub raw_data: Vec<u8>,
}

impl OnnxTensor {
    /// Convert tensor data to f32 values
    pub fn to_f32(&self) -> Vec<f32> {
        match self.data_type {
            OnnxDataType::Float => self
                .raw_data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            OnnxDataType::Float16 => self
                .raw_data
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    f16_to_f32(bits)
                })
                .collect(),
            OnnxDataType::Double => self
                .raw_data
                .chunks_exact(8)
                .map(|b| {
                    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect(),
            OnnxDataType::Int8 => self.raw_data.iter().map(|&b| (b as i8) as f32).collect(),
            OnnxDataType::Uint8 => self.raw_data.iter().map(|&b| b as f32).collect(),
            OnnxDataType::Int32 => self
                .raw_data
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
                .collect(),
            OnnxDataType::Int64 => self
                .raw_data
                .chunks_exact(8)
                .map(|b| {
                    i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// Convert IEEE 754 half-precision to single-precision
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut m = mantissa;
            let mut e = 0u32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            let f32_exp = 127 - 15 - e;
            let f32_mant = (m & 0x3FF) << 13;
            f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant)
        }
    } else if exponent == 31 {
        // Inf/NaN
        let f32_mant = mantissa << 13;
        f32::from_bits((sign << 31) | (0xFF << 23) | f32_mant)
    } else {
        // Normalized
        let f32_exp = exponent + 127 - 15;
        let f32_mant = mantissa << 13;
        f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant)
    }
}

/// ONNX model metadata
#[derive(Debug, Clone, Default)]
pub struct OnnxMetadata {
    /// IR version
    pub ir_version: i64,
    /// Producer name
    pub producer_name: String,
    /// Producer version
    pub producer_version: String,
    /// Domain
    pub domain: String,
    /// Model version
    pub model_version: i64,
    /// Doc string
    pub doc_string: String,
    /// Opset imports
    pub opset_versions: Vec<(String, i64)>,
}

/// ONNX file reader
#[derive(Debug)]
pub struct OnnxReader {
    /// Extracted tensors
    tensors: Vec<OnnxTensor>,
    /// Model metadata
    metadata: OnnxMetadata,
}

include!("reader.rs");
