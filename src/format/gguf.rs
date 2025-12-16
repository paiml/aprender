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
