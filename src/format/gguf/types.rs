//! GGUF types, constants, and serialization (spec §7.2)

use std::collections::BTreeMap;
use std::io::{self, Write};

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
    /// K-quant 4-bit (Q4_K) - 256-element super-blocks, 144 bytes each
    Q4K = 12,
    /// K-quant 6-bit (Q6_K) - 256-element super-blocks, 210 bytes each
    Q6K = 14,
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
        match self.dtype {
            GgmlType::F32 | GgmlType::I32 => elements as usize * 4,
            GgmlType::F16 | GgmlType::I16 => elements as usize * 2,
            GgmlType::I8 => elements as usize,
            GgmlType::Q4_0 | GgmlType::Q4_1 => {
                // Block-quantized: 32 elements per block
                // Q4_0: 2 bytes scale + 16 bytes data = 18 bytes per 32 elements
                ((elements as usize + 31) / 32) * 18
            }
            GgmlType::Q8_0 => {
                // Q8_0: 2 bytes scale + 32 bytes data = 34 bytes per 32 elements
                ((elements as usize + 31) / 32) * 34
            }
            GgmlType::Q4K => {
                // Q4_K: 256-element super-blocks, 144 bytes each
                ((elements as usize + 255) / 256) * 144
            }
            GgmlType::Q6K => {
                // Q6_K: 256-element super-blocks, 210 bytes each
                ((elements as usize + 255) / 256) * 210
            }
            GgmlType::F64 | GgmlType::I64 => elements as usize * 8,
        }
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
///
/// # DEFECT-002 FIX
///
/// Previously, the padding before tensor data was calculated incorrectly, causing
/// the tensor data to be written at the wrong offset. The fix writes header/metadata/
/// tensor-infos to a buffer first to accurately track the byte count for alignment.
pub fn export_tensors_to_gguf<W: Write>(
    writer: &mut W,
    tensors: &[GgufTensor],
    metadata: &[(String, GgufValue)],
) -> Result<()> {
    // DEFECT-002 FIX: Write header section to buffer first to track exact byte count
    let mut header_buffer = Vec::new();

    // Write header
    let header = GgufHeader {
        version: GGUF_VERSION,
        tensor_count: tensors.len() as u64,
        metadata_kv_count: metadata.len() as u64,
    };
    header.write_to(&mut header_buffer)?;

    // Write metadata
    for (key, value) in metadata {
        write_metadata_kv(&mut header_buffer, key, value)?;
    }

    // Calculate tensor data offsets (relative to tensor data section start)
    let mut tensor_data_offset = 0usize;

    // Write tensor infos
    for tensor in tensors {
        let info = GgufTensorInfo {
            name: tensor.name.clone(),
            n_dims: tensor.shape.len() as u32,
            dims: tensor.shape.clone(),
            dtype: tensor.dtype,
            offset: tensor_data_offset as u64,
        };
        info.write_to(&mut header_buffer)?;
        tensor_data_offset += tensor.data.len();
        // Add padding for alignment
        tensor_data_offset += padding_for_alignment(tensor_data_offset, GGUF_DEFAULT_ALIGNMENT);
    }

    // DEFECT-002 FIX: Calculate padding based on actual header buffer size
    let header_size = header_buffer.len();
    let padding = padding_for_alignment(header_size, GGUF_DEFAULT_ALIGNMENT);

    // Write the header buffer to the actual writer
    writer
        .write_all(&header_buffer)
        .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;

    // Write alignment padding before tensor data
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
        let data_padding = padding_for_alignment(tensor.data.len(), GGUF_DEFAULT_ALIGNMENT);
        for _ in 0..data_padding {
            writer
                .write_all(&[0u8])
                .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;
        }
    }

    Ok(())
}

#[cfg(test)]
#[path = "types_tests.rs"]
mod tests;
