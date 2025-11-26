//! GGUF Export (spec §7.2)
//!
//! Pure Rust writer for GGUF format (llama.cpp compatible).
//! WASM compatible - no C/C++ dependencies.
//!
//! # Format Structure
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │ Magic: "GGUF" (4 bytes)                 │
//! │ Version: u32 (currently 3)              │
//! │ Tensor count: u64                       │
//! │ Metadata KV count: u64                  │
//! ├─────────────────────────────────────────┤
//! │ Metadata KV pairs                       │
//! ├─────────────────────────────────────────┤
//! │ Tensor info array                       │
//! ├─────────────────────────────────────────┤
//! │ Tensor data (aligned)                   │
//! └─────────────────────────────────────────┘
//! ```
//!
//! Reference: [GGUF2023] Gerganov, G. (2023). GGUF Format.

use std::io::{self, Write};

use crate::error::{AprenderError, Result};

/// GGUF magic number: "GGUF" in little-endian
pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" as little-endian u32

/// GGUF format version (v3 is current)
pub const GGUF_VERSION: u32 = 3;

/// Default alignment for tensor data
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// GGUF value types (from ggml)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    /// 8-bit unsigned integer
    Uint8 = 0,
    /// 8-bit signed integer
    Int8 = 1,
    /// 16-bit unsigned integer
    Uint16 = 2,
    /// 16-bit signed integer
    Int16 = 3,
    /// 32-bit unsigned integer
    Uint32 = 4,
    /// 32-bit signed integer
    Int32 = 5,
    /// 32-bit float
    Float32 = 6,
    /// Boolean
    Bool = 7,
    /// String (length-prefixed)
    String = 8,
    /// Array of values
    Array = 9,
    /// 64-bit unsigned integer
    Uint64 = 10,
    /// 64-bit signed integer
    Int64 = 11,
    /// 64-bit float
    Float64 = 12,
}

/// GGUF tensor types (from ggml)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    /// 32-bit float
    F32 = 0,
    /// 16-bit float
    F16 = 1,
    /// 4-bit quantization (type 0)
    Q4_0 = 2,
    /// 4-bit quantization (type 1)
    Q4_1 = 3,
    /// 8-bit quantization (type 0)
    Q8_0 = 8,
    /// 8-bit signed integer
    I8 = 24,
    /// 16-bit signed integer
    I16 = 25,
    /// 32-bit signed integer
    I32 = 26,
    /// 64-bit signed integer
    I64 = 27,
    /// 64-bit float
    F64 = 28,
}

/// GGUF metadata value
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    ArrayUint32(Vec<u32>),
    ArrayInt32(Vec<i32>),
    ArrayFloat32(Vec<f32>),
    ArrayString(Vec<String>),
}

impl GgufValue {
    /// Get the GGUF value type
    #[must_use]
    pub const fn value_type(&self) -> GgufValueType {
        match self {
            Self::Uint8(_) => GgufValueType::Uint8,
            Self::Int8(_) => GgufValueType::Int8,
            Self::Uint16(_) => GgufValueType::Uint16,
            Self::Int16(_) => GgufValueType::Int16,
            Self::Uint32(_) => GgufValueType::Uint32,
            Self::Int32(_) => GgufValueType::Int32,
            Self::Float32(_) => GgufValueType::Float32,
            Self::Bool(_) => GgufValueType::Bool,
            Self::String(_) => GgufValueType::String,
            Self::Uint64(_) => GgufValueType::Uint64,
            Self::Int64(_) => GgufValueType::Int64,
            Self::Float64(_) => GgufValueType::Float64,
            Self::ArrayUint32(_)
            | Self::ArrayInt32(_)
            | Self::ArrayFloat32(_)
            | Self::ArrayString(_) => GgufValueType::Array,
        }
    }
}

/// GGUF file header
#[derive(Debug, Clone)]
pub struct GgufHeader {
    /// Format version
    pub version: u32,
    /// Number of tensors in the file
    pub tensor_count: u64,
    /// Number of metadata key-value pairs
    pub metadata_kv_count: u64,
}

impl GgufHeader {
    /// Write the header to a writer
    ///
    /// # Errors
    ///
    /// Returns error on I/O failure
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Magic number
        writer
            .write_all(&GGUF_MAGIC.to_le_bytes())
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;

        // Version
        writer
            .write_all(&self.version.to_le_bytes())
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;

        // Tensor count
        writer
            .write_all(&self.tensor_count.to_le_bytes())
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;

        // Metadata KV count
        writer
            .write_all(&self.metadata_kv_count.to_le_bytes())
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;

        Ok(())
    }
}

/// GGUF tensor info
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name
    pub name: String,
    /// Number of dimensions
    pub n_dims: u32,
    /// Dimensions (shape)
    pub dims: Vec<u64>,
    /// Data type
    pub dtype: GgmlType,
    /// Offset in the data section
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Write tensor info to a writer
    ///
    /// # Errors
    ///
    /// Returns error on I/O failure
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Name (length-prefixed string)
        write_string(writer, &self.name)?;

        // Number of dimensions
        writer
            .write_all(&self.n_dims.to_le_bytes())
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;

        // Dimensions
        for dim in &self.dims {
            writer
                .write_all(&dim.to_le_bytes())
                .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;
        }

        // Data type
        writer
            .write_all(&(self.dtype as u32).to_le_bytes())
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;

        // Offset
        writer
            .write_all(&self.offset.to_le_bytes())
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;

        Ok(())
    }
}

/// Write a length-prefixed string
fn write_string<W: Write>(writer: &mut W, s: &str) -> Result<()> {
    let bytes = s.as_bytes();
    writer
        .write_all(&(bytes.len() as u64).to_le_bytes())
        .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;
    writer
        .write_all(bytes)
        .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;
    Ok(())
}

/// Write a metadata key-value pair
///
/// # Errors
///
/// Returns error on I/O failure
pub fn write_metadata_kv<W: Write>(writer: &mut W, key: &str, value: &GgufValue) -> Result<()> {
    // Key (length-prefixed string)
    write_string(writer, key)?;

    // Value type
    writer
        .write_all(&(value.value_type() as u32).to_le_bytes())
        .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;

    // Value
    write_value(writer, value)?;

    Ok(())
}

/// Write a GGUF value
fn write_value<W: Write>(writer: &mut W, value: &GgufValue) -> Result<()> {
    match value {
        GgufValue::Uint8(v) => writer
            .write_all(&[*v])
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?,
        GgufValue::Int8(v) => writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?,
        GgufValue::Uint16(v) => writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?,
        GgufValue::Int16(v) => writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?,
        GgufValue::Uint32(v) => writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?,
        GgufValue::Int32(v) => writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?,
        GgufValue::Float32(v) => writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?,
        GgufValue::Bool(v) => writer
            .write_all(&[u8::from(*v)])
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?,
        GgufValue::String(v) => write_string(writer, v)?,
        GgufValue::Uint64(v) => writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?,
        GgufValue::Int64(v) => writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?,
        GgufValue::Float64(v) => writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?,
        GgufValue::ArrayUint32(arr) => {
            // Array type + length + elements
            writer
                .write_all(&(GgufValueType::Uint32 as u32).to_le_bytes())
                .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;
            writer
                .write_all(&(arr.len() as u64).to_le_bytes())
                .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;
            for v in arr {
                writer
                    .write_all(&v.to_le_bytes())
                    .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;
            }
        }
        GgufValue::ArrayInt32(arr) => {
            writer
                .write_all(&(GgufValueType::Int32 as u32).to_le_bytes())
                .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;
            writer
                .write_all(&(arr.len() as u64).to_le_bytes())
                .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;
            for v in arr {
                writer
                    .write_all(&v.to_le_bytes())
                    .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;
            }
        }
        GgufValue::ArrayFloat32(arr) => {
            writer
                .write_all(&(GgufValueType::Float32 as u32).to_le_bytes())
                .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;
            writer
                .write_all(&(arr.len() as u64).to_le_bytes())
                .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;
            for v in arr {
                writer
                    .write_all(&v.to_le_bytes())
                    .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;
            }
        }
        GgufValue::ArrayString(arr) => {
            writer
                .write_all(&(GgufValueType::String as u32).to_le_bytes())
                .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;
            writer
                .write_all(&(arr.len() as u64).to_le_bytes())
                .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;
            for s in arr {
                write_string(writer, s)?;
            }
        }
    }
    Ok(())
}

/// Calculate padding bytes needed for alignment
#[must_use]
pub const fn padding_for_alignment(offset: usize, alignment: usize) -> usize {
    let remainder = offset % alignment;
    if remainder == 0 {
        0
    } else {
        alignment - remainder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_constant() {
        assert_eq!(GGUF_MAGIC, 0x4655_4747);
        assert_eq!(&GGUF_MAGIC.to_le_bytes(), b"GGUF");
    }

    #[test]
    fn test_header_size() {
        let mut buffer = Vec::new();
        let header = GgufHeader {
            version: GGUF_VERSION,
            tensor_count: 0,
            metadata_kv_count: 0,
        };
        header.write_to(&mut buffer).expect("write");
        // magic (4) + version (4) + tensor_count (8) + kv_count (8) = 24
        assert_eq!(buffer.len(), 24);
    }

    #[test]
    fn test_padding_calculation() {
        assert_eq!(padding_for_alignment(0, 32), 0);
        assert_eq!(padding_for_alignment(1, 32), 31);
        assert_eq!(padding_for_alignment(32, 32), 0);
        assert_eq!(padding_for_alignment(33, 32), 31);
        assert_eq!(padding_for_alignment(64, 32), 0);
    }

    #[test]
    fn test_value_types() {
        assert_eq!(GgufValue::Uint32(42).value_type(), GgufValueType::Uint32);
        assert_eq!(
            GgufValue::String("test".to_string()).value_type(),
            GgufValueType::String
        );
        assert_eq!(GgufValue::Bool(true).value_type(), GgufValueType::Bool);
    }
}
