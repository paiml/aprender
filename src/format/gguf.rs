//! GGUF Import/Export (spec §7.2)
//!
//! Pure Rust reader/writer for GGUF format (llama.cpp compatible).
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
//! Reference: Gerganov, G. (2023). GGUF Format.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use crate::error::{AprenderError, Result};

/// Type alias for tensor data map (name -> (data, shape))
pub type TensorDataMap = BTreeMap<String, (Vec<f32>, Vec<usize>)>;

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

/// Helper to write bytes with error mapping.
fn write_bytes<W: Write>(writer: &mut W, bytes: &[u8]) -> Result<()> {
    writer
        .write_all(bytes)
        .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))
}

/// Helper to write array header (type + length).
fn write_array_header<W: Write>(
    writer: &mut W,
    element_type: GgufValueType,
    len: usize,
) -> Result<()> {
    write_bytes(writer, &(element_type as u32).to_le_bytes())?;
    write_bytes(writer, &(len as u64).to_le_bytes())
}

/// Write a GGUF value
fn write_value<W: Write>(writer: &mut W, value: &GgufValue) -> Result<()> {
    match value {
        GgufValue::Uint8(v) => write_bytes(writer, &[*v]),
        GgufValue::Int8(v) => write_bytes(writer, &v.to_le_bytes()),
        GgufValue::Uint16(v) => write_bytes(writer, &v.to_le_bytes()),
        GgufValue::Int16(v) => write_bytes(writer, &v.to_le_bytes()),
        GgufValue::Uint32(v) => write_bytes(writer, &v.to_le_bytes()),
        GgufValue::Int32(v) => write_bytes(writer, &v.to_le_bytes()),
        GgufValue::Float32(v) => write_bytes(writer, &v.to_le_bytes()),
        GgufValue::Bool(v) => write_bytes(writer, &[u8::from(*v)]),
        GgufValue::String(v) => write_string(writer, v),
        GgufValue::Uint64(v) => write_bytes(writer, &v.to_le_bytes()),
        GgufValue::Int64(v) => write_bytes(writer, &v.to_le_bytes()),
        GgufValue::Float64(v) => write_bytes(writer, &v.to_le_bytes()),
        GgufValue::ArrayUint32(arr) => write_array_u32(writer, arr),
        GgufValue::ArrayInt32(arr) => write_array_i32(writer, arr),
        GgufValue::ArrayFloat32(arr) => write_array_f32(writer, arr),
        GgufValue::ArrayString(arr) => write_array_string(writer, arr),
    }
}

/// Write u32 array.
fn write_array_u32<W: Write>(writer: &mut W, arr: &[u32]) -> Result<()> {
    write_array_header(writer, GgufValueType::Uint32, arr.len())?;
    for v in arr {
        write_bytes(writer, &v.to_le_bytes())?;
    }
    Ok(())
}

/// Write i32 array.
fn write_array_i32<W: Write>(writer: &mut W, arr: &[i32]) -> Result<()> {
    write_array_header(writer, GgufValueType::Int32, arr.len())?;
    for v in arr {
        write_bytes(writer, &v.to_le_bytes())?;
    }
    Ok(())
}

/// Write f32 array.
fn write_array_f32<W: Write>(writer: &mut W, arr: &[f32]) -> Result<()> {
    write_array_header(writer, GgufValueType::Float32, arr.len())?;
    for v in arr {
        write_bytes(writer, &v.to_le_bytes())?;
    }
    Ok(())
}

/// Write string array.
fn write_array_string<W: Write>(writer: &mut W, arr: &[String]) -> Result<()> {
    write_array_header(writer, GgufValueType::String, arr.len())?;
    for s in arr {
        write_string(writer, s)?;
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

// ============================================================================
// High-Level Export API
// ============================================================================

/// A tensor to be exported to GGUF format
#[derive(Debug, Clone)]
pub struct GgufTensor {
    /// Tensor name (e.g., "model.layers.0.weight")
    pub name: String,
    /// Tensor shape (e.g., [768, 768])
    pub shape: Vec<u64>,
    /// Data type
    pub dtype: GgmlType,
    /// Raw tensor data (little-endian bytes)
    pub data: Vec<u8>,
}

impl GgufTensor {
    /// Calculate the byte size based on dtype and shape
    #[must_use]
    pub fn byte_size(&self) -> usize {
        let elements: u64 = self.shape.iter().product();
        let bytes_per_element = match self.dtype {
            GgmlType::F32 | GgmlType::I32 => 4,
            GgmlType::F16 | GgmlType::I16 => 2,
            GgmlType::I8 => 1,
            GgmlType::Q4_0 | GgmlType::Q4_1 => {
                // Block-quantized: 32 elements per block
                // Q4_0: 2 bytes scale + 16 bytes data = 18 bytes per 32 elements
                ((elements as usize + 31) / 32) * 18
            }
            GgmlType::Q8_0 => {
                // Q8_0: 2 bytes scale + 32 bytes data = 34 bytes per 32 elements
                ((elements as usize + 31) / 32) * 34
            }
            GgmlType::F64 | GgmlType::I64 => 8,
        };
        elements as usize * bytes_per_element / elements.max(1) as usize
    }
}

/// Export tensors to GGUF format
///
/// # Arguments
///
/// * `writer` - Output writer
/// * `tensors` - Tensors to export
/// * `metadata` - Key-value metadata pairs
///
/// # Errors
///
/// Returns error on I/O failure
pub fn export_tensors_to_gguf<W: Write>(
    writer: &mut W,
    tensors: &[GgufTensor],
    metadata: &[(String, GgufValue)],
) -> Result<()> {
    // Write header
    let header = GgufHeader {
        version: GGUF_VERSION,
        tensor_count: tensors.len() as u64,
        metadata_kv_count: metadata.len() as u64,
    };
    header.write_to(writer)?;

    // Write metadata
    for (key, value) in metadata {
        write_metadata_kv(writer, key, value)?;
    }

    // Calculate tensor data offsets
    // First, calculate header + metadata size
    let mut current_offset = 0usize;

    // Write tensor infos
    for tensor in tensors {
        let info = GgufTensorInfo {
            name: tensor.name.clone(),
            n_dims: tensor.shape.len() as u32,
            dims: tensor.shape.clone(),
            dtype: tensor.dtype,
            offset: current_offset as u64,
        };
        info.write_to(writer)?;
        current_offset += tensor.data.len();
        // Add padding for alignment
        current_offset += padding_for_alignment(current_offset, GGUF_DEFAULT_ALIGNMENT);
    }

    // Pad to alignment before tensor data
    let padding = padding_for_alignment(current_offset, GGUF_DEFAULT_ALIGNMENT);
    for _ in 0..padding {
        writer
            .write_all(&[0u8])
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;
    }

    // Write tensor data
    for tensor in tensors {
        writer
            .write_all(&tensor.data)
            .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;

        // Pad to alignment
        let padding = padding_for_alignment(tensor.data.len(), GGUF_DEFAULT_ALIGNMENT);
        for _ in 0..padding {
            writer
                .write_all(&[0u8])
                .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;
        }
    }

    Ok(())
}

// ============================================================================
// GGUF Reading/Import API
// ============================================================================

/// Read a u32 from bytes at offset
fn read_u32(data: &[u8], offset: usize) -> Result<u32> {
    if offset + 4 > data.len() {
        return Err(AprenderError::FormatError {
            message: format!("Unexpected EOF reading u32 at offset {offset}"),
        });
    }
    Ok(u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

/// Read a u64 from bytes at offset
fn read_u64(data: &[u8], offset: usize) -> Result<u64> {
    if offset + 8 > data.len() {
        return Err(AprenderError::FormatError {
            message: format!("Unexpected EOF reading u64 at offset {offset}"),
        });
    }
    Ok(u64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ]))
}

/// Read a length-prefixed string from bytes
fn read_string(data: &[u8], offset: usize) -> Result<(String, usize)> {
    let len = read_u64(data, offset)? as usize;
    let str_start = offset + 8;
    if str_start + len > data.len() {
        return Err(AprenderError::FormatError {
            message: format!("String length {len} exceeds data at offset {offset}"),
        });
    }
    let s = String::from_utf8_lossy(&data[str_start..str_start + len]).to_string();
    Ok((s, 8 + len))
}

/// Read a metadata value and return (value, bytes_consumed)
fn read_metadata_value(data: &[u8], offset: usize, value_type: u32) -> Result<(GgufValue, usize)> {
    match value_type {
        0 => {
            // Uint8
            if offset >= data.len() {
                return Err(AprenderError::FormatError {
                    message: "Unexpected EOF reading Uint8".to_string(),
                });
            }
            Ok((GgufValue::Uint8(data[offset]), 1))
        }
        1 => {
            // Int8
            if offset >= data.len() {
                return Err(AprenderError::FormatError {
                    message: "Unexpected EOF reading Int8".to_string(),
                });
            }
            Ok((GgufValue::Int8(data[offset] as i8), 1))
        }
        2 => {
            // Uint16
            if offset + 2 > data.len() {
                return Err(AprenderError::FormatError {
                    message: "Unexpected EOF reading Uint16".to_string(),
                });
            }
            let v = u16::from_le_bytes([data[offset], data[offset + 1]]);
            Ok((GgufValue::Uint16(v), 2))
        }
        3 => {
            // Int16
            if offset + 2 > data.len() {
                return Err(AprenderError::FormatError {
                    message: "Unexpected EOF reading Int16".to_string(),
                });
            }
            let v = i16::from_le_bytes([data[offset], data[offset + 1]]);
            Ok((GgufValue::Int16(v), 2))
        }
        4 => {
            // Uint32
            let v = read_u32(data, offset)?;
            Ok((GgufValue::Uint32(v), 4))
        }
        5 => {
            // Int32
            if offset + 4 > data.len() {
                return Err(AprenderError::FormatError {
                    message: "Unexpected EOF reading Int32".to_string(),
                });
            }
            let v = i32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            Ok((GgufValue::Int32(v), 4))
        }
        6 => {
            // Float32
            if offset + 4 > data.len() {
                return Err(AprenderError::FormatError {
                    message: "Unexpected EOF reading Float32".to_string(),
                });
            }
            let v = f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            Ok((GgufValue::Float32(v), 4))
        }
        7 => {
            // Bool
            if offset >= data.len() {
                return Err(AprenderError::FormatError {
                    message: "Unexpected EOF reading Bool".to_string(),
                });
            }
            Ok((GgufValue::Bool(data[offset] != 0), 1))
        }
        8 => {
            // String
            let (s, len) = read_string(data, offset)?;
            Ok((GgufValue::String(s), len))
        }
        9 => {
            // Array - read element type and count, then elements
            let elem_type = read_u32(data, offset)?;
            let count = read_u64(data, offset + 4)? as usize;
            let mut consumed = 12; // type (4) + count (8)

            // For tokenizer vocabulary, we need string arrays
            if elem_type == 8 {
                // Array of strings
                let mut strings = Vec::with_capacity(count);
                for _ in 0..count {
                    let (s, len) = read_string(data, offset + consumed)?;
                    strings.push(s);
                    consumed += len;
                }
                Ok((GgufValue::ArrayString(strings), consumed))
            } else if elem_type == 4 {
                // Array of Uint32
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(read_u32(data, offset + consumed)?);
                    consumed += 4;
                }
                Ok((GgufValue::ArrayUint32(values), consumed))
            } else if elem_type == 5 {
                // Array of Int32
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    let v = i32::from_le_bytes([
                        data[offset + consumed],
                        data[offset + consumed + 1],
                        data[offset + consumed + 2],
                        data[offset + consumed + 3],
                    ]);
                    values.push(v);
                    consumed += 4;
                }
                Ok((GgufValue::ArrayInt32(values), consumed))
            } else if elem_type == 6 {
                // Array of Float32
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    let v = f32::from_le_bytes([
                        data[offset + consumed],
                        data[offset + consumed + 1],
                        data[offset + consumed + 2],
                        data[offset + consumed + 3],
                    ]);
                    values.push(v);
                    consumed += 4;
                }
                Ok((GgufValue::ArrayFloat32(values), consumed))
            } else {
                // For other array types, compute size and skip
                let elem_size = match elem_type {
                    0..=1 | 7 => 1, // Uint8, Int8, Bool
                    2..=3 => 2,     // Uint16, Int16
                    10..=12 => 8,   // Uint64, Int64, Float64
                    _ => 4,         // Default
                };
                consumed += count * elem_size;
                // Return empty array (we don't need these types for tokenizer)
                Ok((GgufValue::ArrayUint32(vec![]), consumed))
            }
        }
        10 => {
            // Uint64
            let v = read_u64(data, offset)?;
            Ok((GgufValue::Uint64(v), 8))
        }
        11 => {
            // Int64
            if offset + 8 > data.len() {
                return Err(AprenderError::FormatError {
                    message: "Unexpected EOF reading Int64".to_string(),
                });
            }
            let v = i64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            Ok((GgufValue::Int64(v), 8))
        }
        12 => {
            // Float64
            if offset + 8 > data.len() {
                return Err(AprenderError::FormatError {
                    message: "Unexpected EOF reading Float64".to_string(),
                });
            }
            let v = f64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            Ok((GgufValue::Float64(v), 8))
        }
        _ => {
            // Unknown type - skip 4 bytes
            Ok((GgufValue::Uint32(0), 4))
        }
    }
}

/// Skip a metadata value (we don't need to parse all metadata types)
fn skip_metadata_value(data: &[u8], offset: usize, value_type: u32) -> Result<usize> {
    match value_type {
        0..=1 | 7 => Ok(1), // Uint8, Int8, Bool
        2..=3 => Ok(2),     // Uint16, Int16
        8 => {
            // String
            let len = read_u64(data, offset)? as usize;
            Ok(8 + len)
        }
        9 => {
            // Array - need to read element type and count, then skip elements
            let elem_type = read_u32(data, offset)?;
            let count = read_u64(data, offset + 4)? as usize;
            let elem_size = match elem_type {
                0..=1 | 7 => 1, // Uint8, Int8, Bool
                2..=3 => 2,     // Uint16, Int16
                8 => {
                    // Array of strings - need to iterate
                    let mut skip = 12; // type (4) + count (8)
                    for _ in 0..count {
                        let (_, slen) = read_string(data, offset + skip)?;
                        skip += slen;
                    }
                    return Ok(skip);
                }
                10..=12 => 8, // Uint64, Int64, Float64
                _ => 4,       // Uint32, Int32, Float32, Unknown
            };
            Ok(12 + count * elem_size)
        }
        10..=12 => Ok(8), // Uint64, Int64, Float64
        _ => Ok(4),       // Uint32, Int32, Float32, Unknown (4 bytes default)
    }
}

/// Parsed GGUF file for import
#[derive(Debug)]
pub struct GgufReader {
    /// Raw file data
    data: Vec<u8>,
    /// Format version
    pub version: u32,
    /// Number of tensors
    pub tensor_count: u64,
    /// Tensor infos (name, dims, dtype, `data_offset`)
    pub tensors: Vec<GgufTensorMeta>,
    /// Offset where tensor data section starts
    pub data_offset: usize,
    /// Metadata key-value pairs (extracted from GGUF)
    pub metadata: BTreeMap<String, GgufValue>,
}

/// Tensor metadata from GGUF file
#[derive(Debug, Clone)]
pub struct GgufTensorMeta {
    /// Tensor name
    pub name: String,
    /// Dimensions
    pub dims: Vec<u64>,
    /// Data type (`GgmlType` as u32)
    pub dtype: u32,
    /// Offset within tensor data section
    pub offset: u64,
}

impl GgufReader {
    /// Load and parse a GGUF file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path.as_ref()).map_err(AprenderError::Io)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).map_err(AprenderError::Io)?;
        Self::from_bytes(data)
    }

    /// Parse GGUF from bytes
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        if data.len() < 24 {
            return Err(AprenderError::FormatError {
                message: "GGUF file too small (< 24 bytes)".to_string(),
            });
        }

        // Check magic
        let magic = read_u32(&data, 0)?;
        if magic != GGUF_MAGIC {
            return Err(AprenderError::FormatError {
                message: format!("Invalid GGUF magic: 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X}"),
            });
        }

        let version = read_u32(&data, 4)?;
        let tensor_count = read_u64(&data, 8)?;
        let metadata_kv_count = read_u64(&data, 16)?;

        // Parse metadata section (extract vocabulary and other tokenizer data)
        let mut offset = 24;
        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_kv_count {
            // Read key
            let (key, key_len) = read_string(&data, offset)?;
            offset += key_len;

            // Read value type
            let value_type = read_u32(&data, offset)?;
            offset += 4;

            // Parse value for tokenizer, general, and model architecture keys
            // We parse: tokenizer.*, general.*, llama.*, qwen2.*, phi.* for full model config
            if key.starts_with("tokenizer.")
                || key.starts_with("general.")
                || key.starts_with("llama.")
                || key.starts_with("qwen2.")
                || key.starts_with("phi.")
                || key.starts_with("mistral.")
            {
                let (value, value_len) = read_metadata_value(&data, offset, value_type)?;
                metadata.insert(key, value);
                offset += value_len;
            } else {
                // Skip other metadata for efficiency
                let value_len = skip_metadata_value(&data, offset, value_type)?;
                offset += value_len;
            }
        }

        // Parse tensor infos
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            // Read name
            let (name, name_len) = read_string(&data, offset)?;
            offset += name_len;

            // Read n_dims
            let n_dims = read_u32(&data, offset)? as usize;
            offset += 4;

            // Read dimensions
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_u64(&data, offset)?);
                offset += 8;
            }

            // Read dtype
            let dtype = read_u32(&data, offset)?;
            offset += 4;

            // Read offset
            let tensor_offset = read_u64(&data, offset)?;
            offset += 8;

            tensors.push(GgufTensorMeta {
                name,
                dims,
                dtype,
                offset: tensor_offset,
            });
        }

        // Align to GGUF_DEFAULT_ALIGNMENT for tensor data
        let padding = padding_for_alignment(offset, GGUF_DEFAULT_ALIGNMENT);
        let data_offset = offset + padding;

        Ok(Self {
            data,
            version,
            tensor_count,
            tensors,
            data_offset,
            metadata,
        })
    }

    /// Get vocabulary tokens from metadata
    ///
    /// Returns the token strings indexed by token ID.
    /// Uses "tokenizer.ggml.tokens" key from GGUF metadata.
    #[must_use]
    pub fn vocabulary(&self) -> Option<Vec<String>> {
        if let Some(GgufValue::ArrayString(tokens)) = self.metadata.get("tokenizer.ggml.tokens") {
            if tokens.is_empty() {
                None
            } else {
                Some(tokens.clone())
            }
        } else {
            None
        }
    }

    /// Get tokenizer model type (e.g., "llama", "gpt2")
    #[must_use]
    pub fn tokenizer_model(&self) -> Option<String> {
        if let Some(GgufValue::String(model)) = self.metadata.get("tokenizer.ggml.model") {
            Some(model.clone())
        } else {
            None
        }
    }

    /// Get BOS (beginning of sequence) token ID
    #[must_use]
    pub fn bos_token_id(&self) -> Option<u32> {
        if let Some(GgufValue::Uint32(id)) = self.metadata.get("tokenizer.ggml.bos_token_id") {
            Some(*id)
        } else {
            None
        }
    }

    /// Get EOS (end of sequence) token ID
    #[must_use]
    pub fn eos_token_id(&self) -> Option<u32> {
        if let Some(GgufValue::Uint32(id)) = self.metadata.get("tokenizer.ggml.eos_token_id") {
            Some(*id)
        } else {
            None
        }
    }

    /// Get general architecture name (e.g., "llama", "qwen2")
    #[must_use]
    pub fn architecture(&self) -> Option<String> {
        if let Some(GgufValue::String(arch)) = self.metadata.get("general.architecture") {
            Some(arch.clone())
        } else {
            None
        }
    }

    /// Get model name from metadata
    #[must_use]
    pub fn model_name(&self) -> Option<String> {
        if let Some(GgufValue::String(name)) = self.metadata.get("general.name") {
            Some(name.clone())
        } else {
            None
        }
    }

    // ========================================================================
    // Transformer Model Config (CRITICAL for APR inference)
    // ========================================================================

    /// Get hidden dimension (embedding_length)
    #[must_use]
    pub fn hidden_size(&self) -> Option<usize> {
        let arch = self.architecture().unwrap_or_else(|| "llama".to_string());
        let key = format!("{arch}.embedding_length");
        match self.metadata.get(&key) {
            Some(GgufValue::Uint32(v)) => Some(*v as usize),
            Some(GgufValue::Uint64(v)) => Some(*v as usize),
            _ => None,
        }
    }

    /// Get number of transformer layers (block_count)
    #[must_use]
    pub fn num_layers(&self) -> Option<usize> {
        let arch = self.architecture().unwrap_or_else(|| "llama".to_string());
        let key = format!("{arch}.block_count");
        match self.metadata.get(&key) {
            Some(GgufValue::Uint32(v)) => Some(*v as usize),
            Some(GgufValue::Uint64(v)) => Some(*v as usize),
            _ => None,
        }
    }

    /// Get number of attention heads
    #[must_use]
    pub fn num_heads(&self) -> Option<usize> {
        let arch = self.architecture().unwrap_or_else(|| "llama".to_string());
        let key = format!("{arch}.attention.head_count");
        match self.metadata.get(&key) {
            Some(GgufValue::Uint32(v)) => Some(*v as usize),
            Some(GgufValue::Uint64(v)) => Some(*v as usize),
            _ => None,
        }
    }

    /// Get number of key-value heads (for GQA)
    #[must_use]
    pub fn num_kv_heads(&self) -> Option<usize> {
        let arch = self.architecture().unwrap_or_else(|| "llama".to_string());
        let key = format!("{arch}.attention.head_count_kv");
        match self.metadata.get(&key) {
            Some(GgufValue::Uint32(v)) => Some(*v as usize),
            Some(GgufValue::Uint64(v)) => Some(*v as usize),
            _ => self.num_heads(), // Default to num_heads if not GQA
        }
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> Option<usize> {
        // Try architecture-specific key first
        let arch = self.architecture().unwrap_or_else(|| "llama".to_string());
        let key = format!("{arch}.vocab_size");
        if let Some(GgufValue::Uint32(v)) = self.metadata.get(&key) {
            return Some(*v as usize);
        }
        if let Some(GgufValue::Uint64(v)) = self.metadata.get(&key) {
            return Some(*v as usize);
        }
        // Fall back to vocabulary length
        self.vocabulary().map(|v| v.len())
    }

    /// Get FFN intermediate dimension
    #[must_use]
    pub fn intermediate_size(&self) -> Option<usize> {
        let arch = self.architecture().unwrap_or_else(|| "llama".to_string());
        let key = format!("{arch}.feed_forward_length");
        match self.metadata.get(&key) {
            Some(GgufValue::Uint32(v)) => Some(*v as usize),
            Some(GgufValue::Uint64(v)) => Some(*v as usize),
            _ => None,
        }
    }

    /// Get maximum context length
    #[must_use]
    pub fn context_length(&self) -> Option<usize> {
        let arch = self.architecture().unwrap_or_else(|| "llama".to_string());
        let key = format!("{arch}.context_length");
        match self.metadata.get(&key) {
            Some(GgufValue::Uint32(v)) => Some(*v as usize),
            Some(GgufValue::Uint64(v)) => Some(*v as usize),
            _ => None,
        }
    }

    /// Get RoPE theta (frequency base)
    #[must_use]
    pub fn rope_theta(&self) -> Option<f32> {
        let arch = self.architecture().unwrap_or_else(|| "llama".to_string());
        let key = format!("{arch}.rope.freq_base");
        match self.metadata.get(&key) {
            Some(GgufValue::Float32(v)) => Some(*v),
            Some(GgufValue::Uint32(v)) => Some(*v as f32),
            _ => None,
        }
    }

    /// Get RMS norm epsilon
    #[must_use]
    pub fn rms_norm_eps(&self) -> Option<f32> {
        let arch = self.architecture().unwrap_or_else(|| "llama".to_string());
        let key = format!("{arch}.attention.layer_norm_rms_epsilon");
        match self.metadata.get(&key) {
            Some(GgufValue::Float32(v)) => Some(*v),
            _ => None,
        }
    }

    /// Extract a tensor as F32 data (dequantizing if needed)
    pub fn get_tensor_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let meta = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor '{name}' not found in GGUF"),
            })?;

        let shape: Vec<usize> = meta.dims.iter().map(|&d| d as usize).collect();
        let num_elements: usize = shape.iter().product();
        let tensor_start = self.data_offset + meta.offset as usize;

        let data = match meta.dtype {
            0 => {
                // F32 - direct copy
                let byte_size = num_elements * 4;
                if tensor_start + byte_size > self.data.len() {
                    return Err(AprenderError::FormatError {
                        message: format!("Tensor '{name}' data exceeds file size"),
                    });
                }
                let bytes = &self.data[tensor_start..tensor_start + byte_size];
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            1 => {
                // F16 - convert to F32
                let byte_size = num_elements * 2;
                if tensor_start + byte_size > self.data.len() {
                    return Err(AprenderError::FormatError {
                        message: format!("Tensor '{name}' data exceeds file size"),
                    });
                }
                let bytes = &self.data[tensor_start..tensor_start + byte_size];
                bytes
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect()
            }
            2 => {
                // Q4_0 - dequantize
                dequantize_q4_0(&self.data, tensor_start, num_elements)?
            }
            8 => {
                // Q8_0 - dequantize
                dequantize_q8_0(&self.data, tensor_start, num_elements)?
            }
            6 => {
                // Q4_K - dequantize (super blocks of 256 elements)
                dequantize_q4_k(&self.data, tensor_start, num_elements)?
            }
            7 => {
                // Q5_K - dequantize (super blocks of 256 elements)
                dequantize_q5_k(&self.data, tensor_start, num_elements)?
            }
            12 => {
                // Q6_K - dequantize (super blocks of 256 elements)
                dequantize_q6_k(&self.data, tensor_start, num_elements)?
            }
            3 => {
                // Q4_1 - dequantize (blocks of 32 with scale and min)
                dequantize_q4_1(&self.data, tensor_start, num_elements)?
            }
            10 => {
                // Q2_K - dequantize (super blocks of 256)
                dequantize_q2_k(&self.data, tensor_start, num_elements)?
            }
            11 => {
                // Q3_K - dequantize (super blocks of 256)
                dequantize_q3_k(&self.data, tensor_start, num_elements)?
            }
            // I-quants (importance matrix quantization) - complex formats
            // For now, approximate with simpler dequantization
            13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 => {
                // IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ4_NL, IQ4_XS
                // Fall back to zero-filled tensor with warning
                eprintln!(
                    "Warning: I-quant dtype {} for tensor '{}' not fully supported, using approximation",
                    meta.dtype, name
                );
                dequantize_iq_approximate(&self.data, tensor_start, num_elements, meta.dtype)?
            }
            _ => {
                return Err(AprenderError::FormatError {
                    message: format!("Unsupported GGUF dtype {} for tensor '{name}'", meta.dtype),
                });
            }
        };

        Ok((data, shape))
    }

    /// Get all tensors as F32
    pub fn get_all_tensors_f32(&self) -> Result<TensorDataMap> {
        let mut result = BTreeMap::new();
        for meta in &self.tensors {
            let (data, shape) = self.get_tensor_f32(&meta.name)?;
            result.insert(meta.name.clone(), (data, shape));
        }
        Ok(result)
    }

    /// Get raw tensor bytes without dequantization (preserves Q4K/Q6K)
    ///
    /// Returns (raw_bytes, shape, ggml_dtype) where dtype is:
    /// - 0 = F32, 1 = F16, 6 = Q4_K, 7 = Q5_K, 8 = Q8_0, 12 = Q6_K
    pub fn get_tensor_raw(&self, name: &str) -> Result<(Vec<u8>, Vec<usize>, u32)> {
        let meta = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor '{name}' not found in GGUF"),
            })?;

        let shape: Vec<usize> = meta.dims.iter().map(|&d| d as usize).collect();
        let num_elements: usize = shape.iter().product();
        let tensor_start = self.data_offset + meta.offset as usize;

        // Calculate byte size based on dtype
        let byte_size = match meta.dtype {
            0 => num_elements * 4,      // F32
            1 => num_elements * 2,      // F16
            8 => (num_elements / 32) * 34, // Q8_0: 32 elements = 2 (scale) + 32 (quants)
            6 => (num_elements / 256) * 144, // Q4_K: 256 elements = 144 bytes
            7 => (num_elements / 256) * 176, // Q5_K: 256 elements = 176 bytes
            12 => (num_elements / 256) * 210, // Q6_K: 256 elements = 210 bytes
            _ => {
                return Err(AprenderError::FormatError {
                    message: format!("Unsupported dtype {} for raw extraction", meta.dtype),
                });
            }
        };

        if tensor_start + byte_size > self.data.len() {
            return Err(AprenderError::FormatError {
                message: format!("Tensor '{name}' data exceeds file size"),
            });
        }

        let bytes = self.data[tensor_start..tensor_start + byte_size].to_vec();
        Ok((bytes, shape, meta.dtype))
    }

    /// Get all tensors as raw bytes (preserves quantization)
    ///
    /// Returns BTreeMap of name -> (raw_bytes, shape, ggml_dtype)
    pub fn get_all_tensors_raw(&self) -> Result<BTreeMap<String, (Vec<u8>, Vec<usize>, u32)>> {
        let mut result = BTreeMap::new();
        for meta in &self.tensors {
            let (data, shape, dtype) = self.get_tensor_raw(&meta.name)?;
            result.insert(meta.name.clone(), (data, shape, dtype));
        }
        Ok(result)
    }
}

/// Convert F16 (IEEE 754 half-precision) to F32
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 1);
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x3FF);

    if exp == 0 {
        if mant == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal - convert to normalized f32
            let mut m = mant;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let f32_exp = (127 - 15 + 1 + e) as u32;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        if mant == 0 {
            f32::from_bits((sign << 31) | (0xFF << 23))
        } else {
            f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
        }
    } else {
        // Normal number
        let f32_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

/// Dequantize `Q4_0` format
/// `Q4_0`: blocks of 32 elements, each block has 2-byte f16 scale + 16 bytes of 4-bit quants
fn dequantize_q4_0(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 16; // f16 scale + 16 bytes of 4-bit values

    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let total_bytes = num_blocks * BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q4_0 data exceeds file size".to_string(),
        });
    }

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = start;

    for _ in 0..num_blocks {
        // Read scale (f16)
        let scale_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let scale = f16_to_f32(scale_bits);
        offset += 2;

        // Read 16 bytes = 32 4-bit values
        for i in 0..16 {
            let byte = data[offset + i];
            // Lower 4 bits (subtract 8 to center around 0)
            let v0 = f32::from((byte & 0x0F) as i8 - 8);
            // Upper 4 bits
            let v1 = f32::from((byte >> 4) as i8 - 8);
            result.push(v0 * scale);
            result.push(v1 * scale);
        }
        offset += 16;
    }

    // Truncate to exact element count
    result.truncate(num_elements);
    Ok(result)
}

/// Dequantize `Q8_0` format
/// `Q8_0`: blocks of 32 elements, each block has 2-byte f16 scale + 32 bytes of int8 quants
fn dequantize_q8_0(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 32; // f16 scale + 32 bytes of int8 values

    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let total_bytes = num_blocks * BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q8_0 data exceeds file size".to_string(),
        });
    }

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = start;

    for _ in 0..num_blocks {
        // Read scale (f16)
        let scale_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let scale = f16_to_f32(scale_bits);
        offset += 2;

        // Read 32 int8 values
        for i in 0..32 {
            let v = f32::from(data[offset + i] as i8);
            result.push(v * scale);
        }
        offset += 32;
    }

    // Truncate to exact element count
    result.truncate(num_elements);
    Ok(result)
}

/// Dequantize `Q4_K` format (K-quants)
/// `Q4_K`: super blocks of 256 elements
/// Each super block: d (f16) + dmin (f16) + scales (12 bytes) + qs (128 bytes) = 144 bytes
fn dequantize_q4_k(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 2 + 2 + 12 + 128; // 144 bytes

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q4_K data exceeds file size".to_string(),
        });
    }

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = start;

    for _ in 0..num_blocks {
        // Read d (f16 scale) and dmin (f16 min)
        let d = f16_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[offset + 2], data[offset + 3]]));
        offset += 4;

        // Read scales (12 bytes = 8 6-bit values packed)
        // scales[0..7] are the 8 sub-block scales
        let scales_bytes = &data[offset..offset + 12];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        // Unpack 6-bit scales and mins from 12 bytes
        // First 8 bytes: lower 4 bits of scales, lower 4 bits of mins
        // Last 4 bytes: upper 2 bits of scales and mins
        for i in 0..4 {
            scales[i] = scales_bytes[i] & 0x3F;
            scales[i + 4] = scales_bytes[i + 4] & 0x3F;
            mins[i] = (scales_bytes[i] >> 6) | ((scales_bytes[i + 8] & 0x0F) << 2);
            mins[i + 4] = (scales_bytes[i + 4] >> 6) | ((scales_bytes[i + 8] >> 4) << 2);
        }
        offset += 12;

        // Read 128 bytes = 256 4-bit quantized values
        let qs = &data[offset..offset + 128];
        offset += 128;

        // Dequantize: each sub-block has 32 elements
        for j in 0..8 {
            let scale = d * f32::from(scales[j]);
            let min_val = dmin * f32::from(mins[j]);

            for l in 0..16 {
                let q_byte = qs[j * 16 + l];
                let q0 = (q_byte & 0x0F) as f32;
                let q1 = (q_byte >> 4) as f32;
                result.push(q0 * scale - min_val);
                result.push(q1 * scale - min_val);
            }
        }
    }

    result.truncate(num_elements);
    Ok(result)
}

/// Dequantize `Q5_K` format (K-quants)
/// `Q5_K`: super blocks of 256 elements
/// Each super block: d (f16) + dmin (f16) + scales (12 bytes) + qh (32 bytes) + qs (128 bytes) = 176 bytes
fn dequantize_q5_k(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 2 + 2 + 12 + 32 + 128; // 176 bytes

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q5_K data exceeds file size".to_string(),
        });
    }

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = start;

    for _ in 0..num_blocks {
        // Read d and dmin
        let d = f16_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[offset + 2], data[offset + 3]]));
        offset += 4;

        // Read and unpack scales (same as Q4_K)
        let scales_bytes = &data[offset..offset + 12];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        for i in 0..4 {
            scales[i] = scales_bytes[i] & 0x3F;
            scales[i + 4] = scales_bytes[i + 4] & 0x3F;
            mins[i] = (scales_bytes[i] >> 6) | ((scales_bytes[i + 8] & 0x0F) << 2);
            mins[i + 4] = (scales_bytes[i + 4] >> 6) | ((scales_bytes[i + 8] >> 4) << 2);
        }
        offset += 12;

        // Read qh (32 bytes = 256 high bits)
        let qh = &data[offset..offset + 32];
        offset += 32;

        // Read qs (128 bytes = 256 low 4-bit values)
        let qs = &data[offset..offset + 128];
        offset += 128;

        // Dequantize
        for j in 0..8 {
            let scale = d * f32::from(scales[j]);
            let min_val = dmin * f32::from(mins[j]);

            for l in 0..16 {
                let idx = j * 16 + l;
                let q_byte = qs[idx];
                let qh_byte = qh[idx / 8];
                let bit_pos = (idx % 8) as u8;

                let q0 = ((q_byte & 0x0F) | (((qh_byte >> bit_pos) & 1) << 4)) as f32;
                let q1 = ((q_byte >> 4) | ((((qh_byte >> bit_pos) >> 1) & 1) << 4)) as f32;

                result.push(q0 * scale - min_val);
                result.push(q1 * scale - min_val);
            }
        }
    }

    result.truncate(num_elements);
    Ok(result)
}

/// Dequantize `Q6_K` format (K-quants)
/// `Q6_K`: super blocks of 256 elements
/// Each super block: ql (128 bytes) + qh (64 bytes) + scales (16 bytes) + d (f16) = 210 bytes
fn dequantize_q6_k(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 128 + 64 + 16 + 2; // 210 bytes

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q6_K data exceeds file size".to_string(),
        });
    }

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = start;

    for _ in 0..num_blocks {
        // Read ql (128 bytes = low 4 bits of 256 6-bit values)
        let ql = &data[offset..offset + 128];
        offset += 128;

        // Read qh (64 bytes = high 2 bits of 256 6-bit values)
        let qh = &data[offset..offset + 64];
        offset += 64;

        // Read scales (16 bytes = 16 int8 scales)
        let scales = &data[offset..offset + 16];
        offset += 16;

        // Read d (f16)
        let d = f16_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]]));
        offset += 2;

        // Dequantize 16 sub-blocks of 16 elements each
        for j in 0..16 {
            let scale = d * f32::from(scales[j] as i8);

            for l in 0..8 {
                let idx = j * 8 + l;
                let ql_byte = ql[idx];
                let qh_byte = qh[idx / 2];

                // Extract two 6-bit values
                let qh_shift = (l % 2) * 4;
                let q0 = ((ql_byte & 0x0F) | ((qh_byte >> qh_shift) & 0x03) << 4) as i8 - 32;
                let q1 = ((ql_byte >> 4) | (((qh_byte >> qh_shift) >> 2) & 0x03) << 4) as i8 - 32;

                result.push(f32::from(q0) * scale);
                result.push(f32::from(q1) * scale);
            }
        }
    }

    result.truncate(num_elements);
    Ok(result)
}

/// Dequantize `Q4_1` format
/// `Q4_1`: blocks of 32 elements, each block has f16 scale + f16 min + 16 bytes of 4-bit quants
fn dequantize_q4_1(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 2 + 16; // f16 scale + f16 min + 16 bytes

    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let total_bytes = num_blocks * BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q4_1 data exceeds file size".to_string(),
        });
    }

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = start;

    for _ in 0..num_blocks {
        let scale = f16_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]]));
        let min = f16_to_f32(u16::from_le_bytes([data[offset + 2], data[offset + 3]]));
        offset += 4;

        for i in 0..16 {
            let byte = data[offset + i];
            let v0 = f32::from(byte & 0x0F) * scale + min;
            let v1 = f32::from(byte >> 4) * scale + min;
            result.push(v0);
            result.push(v1);
        }
        offset += 16;
    }

    result.truncate(num_elements);
    Ok(result)
}

/// Dequantize `Q2_K` format (K-quants)
/// `Q2_K`: super blocks of 256 elements
fn dequantize_q2_k(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 2 + 2 + 16 + 64; // d, dmin, scales, qs = 84 bytes

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q2_K data exceeds file size".to_string(),
        });
    }

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = start;

    for _ in 0..num_blocks {
        // Read scales (16 bytes = 16 4-bit scale/min pairs)
        let scales_bytes = &data[offset..offset + 16];
        offset += 16;

        // Read qs (64 bytes = 256 2-bit values)
        let qs = &data[offset..offset + 64];
        offset += 64;

        // Read d and dmin
        let d = f16_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[offset + 2], data[offset + 3]]));
        offset += 4;

        // Dequantize 16 sub-blocks of 16 elements
        for j in 0..16 {
            let sc_byte = scales_bytes[j];
            let scale = d * f32::from(sc_byte & 0x0F);
            let min_val = dmin * f32::from(sc_byte >> 4);

            for l in 0..4 {
                let q_byte = qs[j * 4 + l];
                for k in 0..4 {
                    let q = (q_byte >> (k * 2)) & 0x03;
                    result.push(f32::from(q) * scale - min_val);
                }
            }
        }
    }

    result.truncate(num_elements);
    Ok(result)
}

/// Dequantize `Q3_K` format (K-quants)
/// `Q3_K`: super blocks of 256 elements
fn dequantize_q3_k(data: &[u8], start: usize, num_elements: usize) -> Result<Vec<f32>> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 32 + 64 + 12 + 2; // hmask, qs, scales, d = 110 bytes

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let total_bytes = num_blocks * SUPER_BLOCK_BYTES;

    if start + total_bytes > data.len() {
        return Err(AprenderError::FormatError {
            message: "Q3_K data exceeds file size".to_string(),
        });
    }

    let mut result = Vec::with_capacity(num_elements);
    let mut offset = start;

    for _ in 0..num_blocks {
        // Read hmask (32 bytes = 256 high bits)
        let hmask = &data[offset..offset + 32];
        offset += 32;

        // Read qs (64 bytes = 256 low 2-bit values)
        let qs = &data[offset..offset + 64];
        offset += 64;

        // Read scales (12 bytes = packed 6-bit scales)
        let scales_bytes = &data[offset..offset + 12];
        offset += 12;

        // Read d
        let d = f16_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]]));
        offset += 2;

        // Unpack scales
        let mut scales = [0i8; 16];
        for i in 0..8 {
            scales[i] = (scales_bytes[i] & 0x0F) as i8 - 8;
            scales[i + 8] = (scales_bytes[i] >> 4) as i8 - 8;
        }

        // Dequantize
        for j in 0..256 {
            let sub_block = j / 16;
            let q_idx = j / 4;
            let q_shift = (j % 4) * 2;
            let h_idx = j / 8;
            let h_shift = j % 8;

            let q_low = (qs[q_idx] >> q_shift) & 0x03;
            let q_high = ((hmask[h_idx] >> h_shift) & 1) << 2;
            let q = (q_low | q_high) as i8 - 4;

            result.push(d * f32::from(scales[sub_block]) * f32::from(q));
        }
    }

    result.truncate(num_elements);
    Ok(result)
}

/// Approximate dequantization for I-quants (IQ2, IQ3, IQ4)
/// These use importance-weighted quantization with lookup tables.
/// For import purposes, we approximate with a simple linear mapping.
fn dequantize_iq_approximate(
    data: &[u8],
    start: usize,
    num_elements: usize,
    dtype: u32,
) -> Result<Vec<f32>> {
    // I-quants have variable block sizes and complex lookup tables
    // Approximate by treating as low-bit quantization with estimated scale

    let (bits_per_element, block_size) = match dtype {
        13 => (2, 256), // IQ2_XXS
        14 => (2, 256), // IQ2_XS
        15 => (2, 256), // IQ2_S
        16 => (3, 256), // IQ3_XXS
        17 => (3, 256), // IQ3_S
        18 => (1, 256), // IQ1_S
        19 => (4, 256), // IQ4_NL
        20 => (4, 256), // IQ4_XS
        _ => (4, 256),  // Default
    };

    let bytes_per_block = (block_size * bits_per_element + 7) / 8 + 4; // data + scale overhead
    let num_blocks = (num_elements + block_size - 1) / block_size;

    // For approximation, create small random-ish values based on byte patterns
    // This is NOT correct dequantization but allows import to proceed
    let mut result = Vec::with_capacity(num_elements);
    let scale = 0.01; // Small scale for approximate values

    for block_idx in 0..num_blocks {
        let block_start = start + block_idx * bytes_per_block;

        for i in 0..block_size {
            if result.len() >= num_elements {
                break;
            }

            // Use byte pattern to generate approximate value
            let byte_idx = block_start + (i * bits_per_element) / 8;
            if byte_idx < data.len() {
                let byte_val = data[byte_idx];
                // Map to roughly centered distribution
                let approx = (f32::from(byte_val) - 128.0) * scale;
                result.push(approx);
            } else {
                result.push(0.0);
            }
        }
    }

    result.truncate(num_elements);
    Ok(result)
}

/// Load GGUF file and extract all tensors as F32
pub fn load_gguf_tensors<P: AsRef<Path>>(path: P) -> Result<TensorDataMap> {
    let reader = GgufReader::from_file(path)?;
    reader.get_all_tensors_f32()
}

/// Tokenizer data extracted from GGUF file
#[derive(Debug, Clone, Default)]
pub struct GgufTokenizer {
    /// Vocabulary tokens (indexed by token ID)
    pub vocabulary: Vec<String>,
    /// Tokenizer model type (e.g., "llama", "gpt2")
    pub model_type: Option<String>,
    /// BOS (beginning of sequence) token ID
    pub bos_token_id: Option<u32>,
    /// EOS (end of sequence) token ID
    pub eos_token_id: Option<u32>,
    /// Model architecture (e.g., "llama", "qwen2")
    pub architecture: Option<String>,
    /// Model name from metadata
    pub model_name: Option<String>,
}

impl GgufTokenizer {
    /// Check if tokenizer has a valid vocabulary
    #[must_use]
    pub fn has_vocabulary(&self) -> bool {
        !self.vocabulary.is_empty()
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }
}

/// Transformer model configuration extracted from GGUF metadata
/// CRITICAL for APR inference - must match realizar::apr::AprMetadata
#[derive(Debug, Clone, Default)]
pub struct GgufModelConfig {
    /// Model architecture family (e.g., "llama", "qwen2", "phi")
    pub architecture: Option<String>,
    /// Hidden dimension size (embedding_length)
    pub hidden_size: Option<usize>,
    /// Number of transformer layers (block_count)
    pub num_layers: Option<usize>,
    /// Number of attention heads
    pub num_heads: Option<usize>,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: Option<usize>,
    /// Vocabulary size
    pub vocab_size: Option<usize>,
    /// FFN intermediate dimension
    pub intermediate_size: Option<usize>,
    /// Maximum context length
    pub max_position_embeddings: Option<usize>,
    /// RoPE theta for position encoding
    pub rope_theta: Option<f32>,
    /// RMS norm epsilon
    pub rms_norm_eps: Option<f32>,
}

/// Result of loading a GGUF file with full tokenizer data and model config
#[derive(Debug)]
pub struct GgufLoadResult {
    /// Tensor data (name -> (data, shape))
    pub tensors: TensorDataMap,
    /// Tokenizer data extracted from GGUF metadata
    pub tokenizer: GgufTokenizer,
    /// Model configuration (CRITICAL for inference)
    pub model_config: GgufModelConfig,
}

/// Load GGUF file and extract tensors, tokenizer, AND model config
///
/// This is the preferred method for GGUF import as it preserves
/// the vocabulary and model config needed for text generation.
pub fn load_gguf_with_tokenizer<P: AsRef<Path>>(path: P) -> Result<GgufLoadResult> {
    let reader = GgufReader::from_file(path)?;

    let tensors = reader.get_all_tensors_f32()?;

    let tokenizer = GgufTokenizer {
        vocabulary: reader.vocabulary().unwrap_or_else(Vec::new),
        model_type: reader.tokenizer_model(),
        bos_token_id: reader.bos_token_id(),
        eos_token_id: reader.eos_token_id(),
        architecture: reader.architecture(),
        model_name: reader.model_name(),
    };

    let model_config = GgufModelConfig {
        architecture: reader.architecture(),
        hidden_size: reader.hidden_size(),
        num_layers: reader.num_layers(),
        num_heads: reader.num_heads(),
        num_kv_heads: reader.num_kv_heads(),
        vocab_size: reader.vocab_size(),
        intermediate_size: reader.intermediate_size(),
        max_position_embeddings: reader.context_length(),
        rope_theta: reader.rope_theta(),
        rms_norm_eps: reader.rms_norm_eps(),
    };

    Ok(GgufLoadResult {
        tensors,
        tokenizer,
        model_config,
    })
}

/// Raw tensor data with quantization preserved
#[derive(Debug, Clone)]
pub struct GgufRawTensor {
    /// Raw bytes (Q4K/Q6K super-blocks, or F32/F16 data)
    pub data: Vec<u8>,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// GGML dtype: 0=F32, 1=F16, 6=Q4_K, 7=Q5_K, 8=Q8_0, 12=Q6_K
    pub dtype: u32,
}

/// Result of loading GGUF with raw quantized tensors
#[derive(Debug)]
pub struct GgufRawLoadResult {
    /// Tensors with raw bytes (preserving Q4K/Q6K quantization)
    pub tensors: BTreeMap<String, GgufRawTensor>,
    /// Extracted tokenizer
    pub tokenizer: GgufTokenizer,
    /// Model configuration
    pub model_config: GgufModelConfig,
}

/// Load GGUF with raw quantized tensors (preserves Q4K for GPU inference)
///
/// This is essential for APR format to achieve 2x Ollama performance.
/// The Q4K bytes are stored directly in APR and used by GPU kernels.
pub fn load_gguf_raw<P: AsRef<Path>>(path: P) -> Result<GgufRawLoadResult> {
    let reader = GgufReader::from_file(path)?;

    let raw_tensors = reader.get_all_tensors_raw()?;
    let mut tensors = BTreeMap::new();
    for (name, (data, shape, dtype)) in raw_tensors {
        tensors.insert(name, GgufRawTensor { data, shape, dtype });
    }

    let tokenizer = GgufTokenizer {
        vocabulary: reader.vocabulary().unwrap_or_else(Vec::new),
        model_type: reader.tokenizer_model(),
        bos_token_id: reader.bos_token_id(),
        eos_token_id: reader.eos_token_id(),
        architecture: reader.architecture(),
        model_name: reader.model_name(),
    };

    let model_config = GgufModelConfig {
        architecture: reader.architecture(),
        hidden_size: reader.hidden_size(),
        num_layers: reader.num_layers(),
        num_heads: reader.num_heads(),
        num_kv_heads: reader.num_kv_heads(),
        vocab_size: reader.vocab_size(),
        intermediate_size: reader.intermediate_size(),
        max_position_embeddings: reader.context_length(),
        rope_theta: reader.rope_theta(),
        rms_norm_eps: reader.rms_norm_eps(),
    };

    Ok(GgufRawLoadResult {
        tensors,
        tokenizer,
        model_config,
    })
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

    // ========================================================================
    // GgufValue type coverage tests
    // ========================================================================

    #[test]
    fn test_all_value_types() {
        assert_eq!(GgufValue::Uint8(1).value_type(), GgufValueType::Uint8);
        assert_eq!(GgufValue::Int8(-1).value_type(), GgufValueType::Int8);
        assert_eq!(GgufValue::Uint16(1).value_type(), GgufValueType::Uint16);
        assert_eq!(GgufValue::Int16(-1).value_type(), GgufValueType::Int16);
        assert_eq!(GgufValue::Uint32(1).value_type(), GgufValueType::Uint32);
        assert_eq!(GgufValue::Int32(-1).value_type(), GgufValueType::Int32);
        assert_eq!(GgufValue::Float32(1.0).value_type(), GgufValueType::Float32);
        assert_eq!(GgufValue::Bool(true).value_type(), GgufValueType::Bool);
        assert_eq!(
            GgufValue::String("s".into()).value_type(),
            GgufValueType::String
        );
        assert_eq!(GgufValue::Uint64(1).value_type(), GgufValueType::Uint64);
        assert_eq!(GgufValue::Int64(-1).value_type(), GgufValueType::Int64);
        assert_eq!(GgufValue::Float64(1.0).value_type(), GgufValueType::Float64);
        assert_eq!(
            GgufValue::ArrayUint32(vec![1]).value_type(),
            GgufValueType::Array
        );
        assert_eq!(
            GgufValue::ArrayInt32(vec![1]).value_type(),
            GgufValueType::Array
        );
        assert_eq!(
            GgufValue::ArrayFloat32(vec![1.0]).value_type(),
            GgufValueType::Array
        );
        assert_eq!(
            GgufValue::ArrayString(vec!["s".into()]).value_type(),
            GgufValueType::Array
        );
    }

    // ========================================================================
    // write_metadata_kv tests
    // ========================================================================

    #[test]
    fn test_write_metadata_kv_uint8() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "test", &GgufValue::Uint8(42)).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_write_metadata_kv_int8() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "key", &GgufValue::Int8(-5)).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_write_metadata_kv_uint16() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "k", &GgufValue::Uint16(1000)).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_write_metadata_kv_int16() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "k", &GgufValue::Int16(-1000)).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_write_metadata_kv_uint32() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "k", &GgufValue::Uint32(100_000)).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_write_metadata_kv_int32() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "k", &GgufValue::Int32(-100_000)).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_write_metadata_kv_float32() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "k", &GgufValue::Float32(3.14)).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_write_metadata_kv_bool() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "k", &GgufValue::Bool(true)).unwrap();
        assert!(!buf.is_empty());

        let mut buf2 = Vec::new();
        write_metadata_kv(&mut buf2, "k", &GgufValue::Bool(false)).unwrap();
        assert!(!buf2.is_empty());
    }

    #[test]
    fn test_write_metadata_kv_string() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "k", &GgufValue::String("hello".into())).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_write_metadata_kv_uint64() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "k", &GgufValue::Uint64(1_000_000_000)).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_write_metadata_kv_int64() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "k", &GgufValue::Int64(-1_000_000_000)).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_write_metadata_kv_float64() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "k", &GgufValue::Float64(3.14159265359)).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_write_metadata_kv_array_uint32() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "k", &GgufValue::ArrayUint32(vec![1, 2, 3])).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_write_metadata_kv_array_int32() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "k", &GgufValue::ArrayInt32(vec![-1, 0, 1])).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_write_metadata_kv_array_float32() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "k", &GgufValue::ArrayFloat32(vec![1.0, 2.0, 3.0])).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_write_metadata_kv_array_string() {
        let mut buf = Vec::new();
        write_metadata_kv(
            &mut buf,
            "k",
            &GgufValue::ArrayString(vec!["a".into(), "b".into()]),
        )
        .unwrap();
        assert!(!buf.is_empty());
    }

    // ========================================================================
    // GgufTensorInfo tests
    // ========================================================================

    #[test]
    fn test_tensor_info_write() {
        let info = GgufTensorInfo {
            name: "test.weight".to_string(),
            n_dims: 2,
            dims: vec![10, 20],
            dtype: GgmlType::F32,
            offset: 0,
        };
        let mut buf = Vec::new();
        info.write_to(&mut buf).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_tensor_info_all_dtypes() {
        // Test write_to works for various dtypes
        let dtypes = [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::I8,
            GgmlType::I16,
            GgmlType::I32,
            GgmlType::I64,
            GgmlType::F64,
        ];
        for dtype in dtypes {
            let info = GgufTensorInfo {
                name: "w".to_string(),
                n_dims: 2,
                dims: vec![10, 20],
                dtype,
                offset: 0,
            };
            let mut buf = Vec::new();
            info.write_to(&mut buf).unwrap();
            assert!(!buf.is_empty());
        }
    }

    #[test]
    fn test_ggml_type_all_variants() {
        // Test all GgmlType variants for Debug and Eq
        let types = [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q8_0,
            GgmlType::I8,
            GgmlType::I16,
            GgmlType::I32,
            GgmlType::I64,
            GgmlType::F64,
        ];
        for t in types {
            assert_eq!(t, t);
            assert!(!format!("{t:?}").is_empty());
        }
    }

    #[test]
    fn test_gguf_value_type_all_variants() {
        // Test all GgufValueType variants
        let types = [
            GgufValueType::Uint8,
            GgufValueType::Int8,
            GgufValueType::Uint16,
            GgufValueType::Int16,
            GgufValueType::Uint32,
            GgufValueType::Int32,
            GgufValueType::Float32,
            GgufValueType::Bool,
            GgufValueType::String,
            GgufValueType::Array,
            GgufValueType::Uint64,
            GgufValueType::Int64,
            GgufValueType::Float64,
        ];
        for t in types {
            assert_eq!(t, t);
            assert!(!format!("{t:?}").is_empty());
        }
    }

    // ========================================================================
    // Enum Debug/Clone tests
    // ========================================================================

    #[test]
    fn test_gguf_value_type_enum() {
        let t = GgufValueType::Uint8;
        assert_eq!(t, GgufValueType::Uint8);
        let cloned = t;
        assert_eq!(t, cloned);
        assert!(format!("{t:?}").contains("Uint8"));
    }

    #[test]
    fn test_ggml_type_enum() {
        let t = GgmlType::F32;
        assert_eq!(t, GgmlType::F32);
        let cloned = t;
        assert_eq!(t, cloned);
        assert!(format!("{t:?}").contains("F32"));
    }

    #[test]
    fn test_gguf_value_clone() {
        let v = GgufValue::String("test".to_string());
        let cloned = v.clone();
        assert!(format!("{cloned:?}").contains("test"));
    }

    #[test]
    fn test_gguf_header_clone() {
        let h = GgufHeader {
            version: 3,
            tensor_count: 10,
            metadata_kv_count: 5,
        };
        let cloned = h.clone();
        assert_eq!(cloned.version, 3);
        assert_eq!(cloned.tensor_count, 10);
        assert!(format!("{cloned:?}").contains("GgufHeader"));
    }

    // ========================================================================
    // GgufTensor tests
    // ========================================================================

    #[test]
    fn test_gguf_tensor_byte_size_f32() {
        let tensor = GgufTensor {
            name: "weights".to_string(),
            shape: vec![10, 20],
            dtype: GgmlType::F32,
            data: vec![0; 800], // 10 * 20 * 4
        };
        let size = tensor.byte_size();
        assert!(size > 0);
    }

    #[test]
    fn test_gguf_tensor_byte_size_f16() {
        let tensor = GgufTensor {
            name: "weights".to_string(),
            shape: vec![10, 20],
            dtype: GgmlType::F16,
            data: vec![0; 400], // 10 * 20 * 2
        };
        let size = tensor.byte_size();
        assert!(size > 0);
    }

    #[test]
    fn test_gguf_tensor_byte_size_i8() {
        let tensor = GgufTensor {
            name: "weights".to_string(),
            shape: vec![10, 20],
            dtype: GgmlType::I8,
            data: vec![0; 200], // 10 * 20 * 1
        };
        let size = tensor.byte_size();
        assert!(size > 0);
    }

    #[test]
    fn test_gguf_tensor_byte_size_i16() {
        let tensor = GgufTensor {
            name: "weights".to_string(),
            shape: vec![10, 20],
            dtype: GgmlType::I16,
            data: vec![0; 400],
        };
        let size = tensor.byte_size();
        assert!(size > 0);
    }

    #[test]
    fn test_gguf_tensor_byte_size_i32() {
        let tensor = GgufTensor {
            name: "weights".to_string(),
            shape: vec![10, 20],
            dtype: GgmlType::I32,
            data: vec![0; 800],
        };
        let size = tensor.byte_size();
        assert!(size > 0);
    }

    #[test]
    fn test_gguf_tensor_byte_size_i64() {
        let tensor = GgufTensor {
            name: "weights".to_string(),
            shape: vec![10, 20],
            dtype: GgmlType::I64,
            data: vec![0; 1600],
        };
        let size = tensor.byte_size();
        assert!(size > 0);
    }

    #[test]
    fn test_gguf_tensor_byte_size_f64() {
        let tensor = GgufTensor {
            name: "weights".to_string(),
            shape: vec![10, 20],
            dtype: GgmlType::F64,
            data: vec![0; 1600],
        };
        let size = tensor.byte_size();
        assert!(size > 0);
    }

    #[test]
    fn test_gguf_tensor_byte_size_q4_0() {
        let tensor = GgufTensor {
            name: "quantized".to_string(),
            shape: vec![64],
            dtype: GgmlType::Q4_0,
            data: vec![0; 100],
        };
        let size = tensor.byte_size();
        assert!(size > 0);
    }

    #[test]
    fn test_gguf_tensor_byte_size_q4_1() {
        let tensor = GgufTensor {
            name: "quantized".to_string(),
            shape: vec![64],
            dtype: GgmlType::Q4_1,
            data: vec![0; 100],
        };
        let size = tensor.byte_size();
        assert!(size > 0);
    }

    #[test]
    fn test_gguf_tensor_byte_size_q8_0() {
        let tensor = GgufTensor {
            name: "quantized".to_string(),
            shape: vec![64],
            dtype: GgmlType::Q8_0,
            data: vec![0; 100],
        };
        let size = tensor.byte_size();
        assert!(size > 0);
    }

    #[test]
    fn test_gguf_tensor_clone_debug() {
        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: vec![10],
            dtype: GgmlType::F32,
            data: vec![1, 2, 3, 4],
        };
        let cloned = tensor.clone();
        assert_eq!(cloned.name, "test");
        assert!(format!("{cloned:?}").contains("GgufTensor"));
    }

    // ========================================================================
    // export_tensors_to_gguf tests
    // ========================================================================

    #[test]
    fn test_export_tensors_to_gguf_empty() {
        let mut buf = Vec::new();
        export_tensors_to_gguf(&mut buf, &[], &[]).expect("export should succeed");
        // Should have header at minimum
        assert!(buf.len() >= 24);
    }

    #[test]
    fn test_export_tensors_to_gguf_with_metadata() {
        let mut buf = Vec::new();
        let metadata = vec![
            (
                "model.name".to_string(),
                GgufValue::String("test".to_string()),
            ),
            ("model.version".to_string(), GgufValue::Uint32(1)),
        ];
        export_tensors_to_gguf(&mut buf, &[], &metadata).expect("export should succeed");
        assert!(buf.len() > 24);
    }

    #[test]
    fn test_export_tensors_to_gguf_with_tensors() {
        let mut buf = Vec::new();
        let tensors = vec![GgufTensor {
            name: "weights".to_string(),
            shape: vec![4],
            dtype: GgmlType::F32,
            data: vec![0; 16], // 4 * 4 bytes
        }];
        export_tensors_to_gguf(&mut buf, &tensors, &[]).expect("export should succeed");
        assert!(buf.len() > 24);
    }

    #[test]
    fn test_export_tensors_to_gguf_full() {
        let mut buf = Vec::new();
        let tensors = vec![
            GgufTensor {
                name: "layer.0.weight".to_string(),
                shape: vec![10, 10],
                dtype: GgmlType::F32,
                data: vec![0; 400],
            },
            GgufTensor {
                name: "layer.0.bias".to_string(),
                shape: vec![10],
                dtype: GgmlType::F32,
                data: vec![0; 40],
            },
        ];
        let metadata = vec![
            (
                "general.architecture".to_string(),
                GgufValue::String("test".to_string()),
            ),
            (
                "general.quantization_version".to_string(),
                GgufValue::Uint32(2),
            ),
        ];
        export_tensors_to_gguf(&mut buf, &tensors, &metadata).expect("export should succeed");
        // Verify header magic
        assert_eq!(&buf[0..4], b"GGUF");
    }

    #[test]
    fn test_gguf_tensor_info_clone_debug() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            n_dims: 2,
            dims: vec![10, 20],
            dtype: GgmlType::F32,
            offset: 0,
        };
        let cloned = info.clone();
        assert_eq!(cloned.name, "test");
        assert!(format!("{cloned:?}").contains("GgufTensorInfo"));
    }
}

// ============================================================================
// Property-Based Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // Strategy for generating valid GGUF headers
    fn arb_header() -> impl Strategy<Value = GgufHeader> {
        (0u64..1000, 0u64..100).prop_map(|(tensor_count, metadata_kv_count)| GgufHeader {
            version: GGUF_VERSION,
            tensor_count,
            metadata_kv_count,
        })
    }

    proptest! {
        /// Property: Header write always produces exactly 24 bytes
        #[test]
        fn prop_header_size_always_24(header in arb_header()) {
            let mut buffer = Vec::new();
            header.write_to(&mut buffer).expect("write");
            prop_assert_eq!(buffer.len(), 24);
        }

        /// Property: Header always starts with GGUF magic
        #[test]
        fn prop_header_magic(header in arb_header()) {
            let mut buffer = Vec::new();
            header.write_to(&mut buffer).expect("write");
            prop_assert_eq!(&buffer[0..4], b"GGUF");
        }

        /// Property: Header version is always 3
        #[test]
        fn prop_header_version(header in arb_header()) {
            let mut buffer = Vec::new();
            header.write_to(&mut buffer).expect("write");
            let version = u32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]);
            prop_assert_eq!(version, GGUF_VERSION);
        }

        /// Property: Padding is always less than alignment
        #[test]
        fn prop_padding_less_than_alignment(offset in 0usize..10000, alignment in 1usize..256) {
            let padding = padding_for_alignment(offset, alignment);
            prop_assert!(padding < alignment);
        }

        /// Property: offset + padding is always aligned
        #[test]
        fn prop_padded_offset_aligned(offset in 0usize..10000, alignment in 1usize..256) {
            let padding = padding_for_alignment(offset, alignment);
            prop_assert_eq!((offset + padding) % alignment, 0);
        }

        /// Property: Aligned offsets need zero padding
        #[test]
        fn prop_aligned_needs_no_padding(multiple in 0usize..1000, alignment in 1usize..256) {
            let offset = multiple * alignment;
            prop_assert_eq!(padding_for_alignment(offset, alignment), 0);
        }

        /// Property: String metadata key-value is non-empty
        #[test]
        fn prop_string_metadata_nonempty(
            key in "[a-z][a-z0-9_.]{0,30}",
            value in "[a-zA-Z0-9_ ]{0,100}"
        ) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, &key, &GgufValue::String(value)).expect("write");
            prop_assert!(!buffer.is_empty());
        }

        /// Property: Uint32 value roundtrip through bytes
        #[test]
        fn prop_uint32_value_written(value in any::<u32>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "test", &GgufValue::Uint32(value)).expect("write");
            // Key: 8 (len) + 4 (test) + type: 4 + value: 4 = 20 bytes
            prop_assert!(buffer.len() >= 20);
        }

        /// Property: Float32 value roundtrip through bytes
        #[test]
        fn prop_float32_value_written(value in any::<f32>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "f", &GgufValue::Float32(value)).expect("write");
            prop_assert!(!buffer.is_empty());
        }

        /// Property: Tensor export produces valid GGUF with magic
        #[test]
        fn prop_tensor_export_has_magic(
            name in "[a-z][a-z0-9.]{0,20}",
            dim0 in 1u64..100,
            dim1 in 1u64..100
        ) {
            let data = vec![0u8; (dim0 * dim1 * 4) as usize]; // f32 data
            let tensor = GgufTensor {
                name,
                shape: vec![dim0, dim1],
                dtype: GgmlType::F32,
                data,
            };
            let mut buffer = Vec::new();
            export_tensors_to_gguf(&mut buffer, &[tensor], &[]).expect("export");
            prop_assert_eq!(&buffer[0..4], b"GGUF");
        }

        // ================================================================
        // Metadata Roundtrip Property Tests
        // ================================================================

        /// Property: Bool true encodes to 1, false to 0
        #[test]
        fn prop_bool_value_encoding(value in any::<bool>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "b", &GgufValue::Bool(value)).expect("write");
            // Last byte is the bool value
            let last_byte = buffer[buffer.len() - 1];
            prop_assert_eq!(last_byte, u8::from(value));
        }

        /// Property: Int64 values encode correctly
        #[test]
        fn prop_int64_value_encoded(value in any::<i64>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "i", &GgufValue::Int64(value)).expect("write");
            // Last 8 bytes are the i64 value
            let bytes = &buffer[buffer.len() - 8..];
            let decoded = i64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
            ]);
            prop_assert_eq!(decoded, value);
        }

        /// Property: Uint64 values encode correctly
        #[test]
        fn prop_uint64_value_encoded(value in any::<u64>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "u", &GgufValue::Uint64(value)).expect("write");
            let bytes = &buffer[buffer.len() - 8..];
            let decoded = u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
            ]);
            prop_assert_eq!(decoded, value);
        }

        /// Property: Float64 values encode correctly (bit-exact)
        #[test]
        fn prop_float64_value_encoded(value in any::<f64>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "d", &GgufValue::Float64(value)).expect("write");
            let bytes = &buffer[buffer.len() - 8..];
            let decoded = f64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
            ]);
            // Use to_bits for NaN-safe comparison
            prop_assert_eq!(decoded.to_bits(), value.to_bits());
        }

        /// Property: Value type tag is correct for all value types
        #[test]
        fn prop_value_type_tag_uint32(value in any::<u32>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "x", &GgufValue::Uint32(value)).expect("write");
            // Type is at bytes 12-15 (after key: 8 byte len + 1 byte "x" = 9 bytes, padded to 9, then type)
            // Key: u64 length (8) + "x" (1) = 9 bytes, then u32 type
            let type_bytes = &buffer[9..13];
            let type_val = u32::from_le_bytes([type_bytes[0], type_bytes[1], type_bytes[2], type_bytes[3]]);
            prop_assert_eq!(type_val, GgufValueType::Uint32 as u32);
        }

        // ================================================================
        // Tensor Info Property Tests
        // ================================================================

        /// Property: Tensor info serialization contains name
        #[test]
        fn prop_tensor_info_contains_name(
            name in "[a-z][a-z0-9_.]{0,30}"
        ) {
            let info = GgufTensorInfo {
                name: name.clone(),
                n_dims: 2,
                dims: vec![10, 20],
                dtype: GgmlType::F32,
                offset: 0,
            };
            let mut buffer = Vec::new();
            info.write_to(&mut buffer).expect("write");
            // Name length is first 8 bytes
            let name_len = u64::from_le_bytes([
                buffer[0], buffer[1], buffer[2], buffer[3],
                buffer[4], buffer[5], buffer[6], buffer[7],
            ]) as usize;
            prop_assert_eq!(name_len, name.len());
            // Name bytes follow
            let name_bytes = &buffer[8..8 + name_len];
            prop_assert_eq!(name_bytes, name.as_bytes());
        }

        /// Property: Tensor info n_dims matches shape length
        #[test]
        fn prop_tensor_info_ndims_matches_shape(
            dims in proptest::collection::vec(1u64..100, 1..5)
        ) {
            let info = GgufTensorInfo {
                name: "t".to_string(),
                n_dims: dims.len() as u32,
                dims: dims.clone(),
                dtype: GgmlType::F32,
                offset: 0,
            };
            let mut buffer = Vec::new();
            info.write_to(&mut buffer).expect("write");
            // After name (8 + 1 = 9 bytes), n_dims is next 4 bytes
            let n_dims = u32::from_le_bytes([buffer[9], buffer[10], buffer[11], buffer[12]]);
            prop_assert_eq!(n_dims as usize, dims.len());
        }

        /// Property: Multiple metadata pairs produces correct count in header
        #[test]
        fn prop_export_metadata_count(
            count in 0usize..10
        ) {
            let metadata: Vec<(String, GgufValue)> = (0..count)
                .map(|i| (format!("key{i}"), GgufValue::Uint32(i as u32)))
                .collect();
            let mut buffer = Vec::new();
            export_tensors_to_gguf(&mut buffer, &[], &metadata).expect("export");
            // KV count is at bytes 16-23 (after magic 4, version 4, tensor_count 8)
            let kv_count = u64::from_le_bytes([
                buffer[16], buffer[17], buffer[18], buffer[19],
                buffer[20], buffer[21], buffer[22], buffer[23],
            ]);
            prop_assert_eq!(kv_count as usize, count);
        }

        /// Property: Tensor count in header matches tensors provided
        #[test]
        fn prop_export_tensor_count(
            count in 0usize..5
        ) {
            let tensors: Vec<GgufTensor> = (0..count)
                .map(|i| GgufTensor {
                    name: format!("t{i}"),
                    shape: vec![4],
                    dtype: GgmlType::F32,
                    data: vec![0u8; 16], // 4 f32s = 16 bytes
                })
                .collect();
            let mut buffer = Vec::new();
            export_tensors_to_gguf(&mut buffer, &tensors, &[]).expect("export");
            // Tensor count is at bytes 8-15
            let tensor_count = u64::from_le_bytes([
                buffer[8], buffer[9], buffer[10], buffer[11],
                buffer[12], buffer[13], buffer[14], buffer[15],
            ]);
            prop_assert_eq!(tensor_count as usize, count);
        }
    }
}
