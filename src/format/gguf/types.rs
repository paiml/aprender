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
mod tests {
    use super::*;

    // ============================================================================
    // Constants Tests
    // ============================================================================

    #[test]
    fn test_gguf_magic_is_correct() {
        // "GGUF" in little-endian
        assert_eq!(GGUF_MAGIC, 0x4655_4747);
        let bytes = GGUF_MAGIC.to_le_bytes();
        assert_eq!(&bytes, b"GGUF");
    }

    #[test]
    fn test_gguf_version_is_v3() {
        assert_eq!(GGUF_VERSION, 3);
    }

    #[test]
    fn test_gguf_default_alignment() {
        assert_eq!(GGUF_DEFAULT_ALIGNMENT, 32);
    }

    // ============================================================================
    // GgufValueType Tests
    // ============================================================================

    #[test]
    fn test_gguf_value_type_discriminants() {
        assert_eq!(GgufValueType::Uint8 as u32, 0);
        assert_eq!(GgufValueType::Int8 as u32, 1);
        assert_eq!(GgufValueType::Uint16 as u32, 2);
        assert_eq!(GgufValueType::Int16 as u32, 3);
        assert_eq!(GgufValueType::Uint32 as u32, 4);
        assert_eq!(GgufValueType::Int32 as u32, 5);
        assert_eq!(GgufValueType::Float32 as u32, 6);
        assert_eq!(GgufValueType::Bool as u32, 7);
        assert_eq!(GgufValueType::String as u32, 8);
        assert_eq!(GgufValueType::Array as u32, 9);
        assert_eq!(GgufValueType::Uint64 as u32, 10);
        assert_eq!(GgufValueType::Int64 as u32, 11);
        assert_eq!(GgufValueType::Float64 as u32, 12);
    }

    #[test]
    fn test_gguf_value_type_clone_eq() {
        let t1 = GgufValueType::Float32;
        let t2 = t1;
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_gguf_value_type_debug() {
        let s = format!("{:?}", GgufValueType::String);
        assert!(s.contains("String"));
    }

    // ============================================================================
    // GgmlType Tests
    // ============================================================================

    #[test]
    fn test_ggml_type_discriminants() {
        assert_eq!(GgmlType::F32 as u32, 0);
        assert_eq!(GgmlType::F16 as u32, 1);
        assert_eq!(GgmlType::Q4_0 as u32, 2);
        assert_eq!(GgmlType::Q4_1 as u32, 3);
        assert_eq!(GgmlType::Q8_0 as u32, 8);
        assert_eq!(GgmlType::Q4K as u32, 12);
        assert_eq!(GgmlType::Q6K as u32, 14);
        assert_eq!(GgmlType::I8 as u32, 24);
        assert_eq!(GgmlType::I16 as u32, 25);
        assert_eq!(GgmlType::I32 as u32, 26);
        assert_eq!(GgmlType::I64 as u32, 27);
        assert_eq!(GgmlType::F64 as u32, 28);
    }

    #[test]
    fn test_ggml_type_clone_eq() {
        let t1 = GgmlType::Q4K;
        let t2 = t1;
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_ggml_type_debug() {
        let s = format!("{:?}", GgmlType::Q6K);
        assert!(s.contains("Q6K"));
    }

    // ============================================================================
    // GgufValue Tests
    // ============================================================================

    #[test]
    fn test_gguf_value_type_mapping() {
        assert_eq!(GgufValue::Uint8(42).value_type(), GgufValueType::Uint8);
        assert_eq!(GgufValue::Int8(-1).value_type(), GgufValueType::Int8);
        assert_eq!(GgufValue::Uint16(1000).value_type(), GgufValueType::Uint16);
        assert_eq!(GgufValue::Int16(-500).value_type(), GgufValueType::Int16);
        assert_eq!(GgufValue::Uint32(100_000).value_type(), GgufValueType::Uint32);
        assert_eq!(GgufValue::Int32(-100_000).value_type(), GgufValueType::Int32);
        assert_eq!(GgufValue::Float32(3.14).value_type(), GgufValueType::Float32);
        assert_eq!(GgufValue::Bool(true).value_type(), GgufValueType::Bool);
        assert_eq!(
            GgufValue::String("test".to_string()).value_type(),
            GgufValueType::String
        );
        assert_eq!(GgufValue::Uint64(u64::MAX).value_type(), GgufValueType::Uint64);
        assert_eq!(GgufValue::Int64(i64::MIN).value_type(), GgufValueType::Int64);
        assert_eq!(GgufValue::Float64(2.718).value_type(), GgufValueType::Float64);
    }

    #[test]
    fn test_gguf_value_array_types() {
        assert_eq!(
            GgufValue::ArrayUint32(vec![1, 2, 3]).value_type(),
            GgufValueType::Array
        );
        assert_eq!(
            GgufValue::ArrayInt32(vec![-1, 0, 1]).value_type(),
            GgufValueType::Array
        );
        assert_eq!(
            GgufValue::ArrayFloat32(vec![1.0, 2.0]).value_type(),
            GgufValueType::Array
        );
        assert_eq!(
            GgufValue::ArrayString(vec!["a".to_string()]).value_type(),
            GgufValueType::Array
        );
    }

    #[test]
    fn test_gguf_value_clone() {
        let v1 = GgufValue::String("hello".to_string());
        let v2 = v1.clone();
        if let (GgufValue::String(s1), GgufValue::String(s2)) = (&v1, &v2) {
            assert_eq!(s1, s2);
        } else {
            panic!("Clone failed");
        }
    }

    #[test]
    fn test_gguf_value_debug() {
        let v = GgufValue::Float32(1.5);
        let s = format!("{v:?}");
        assert!(s.contains("Float32"));
        assert!(s.contains("1.5"));
    }

    // ============================================================================
    // GgufHeader Tests
    // ============================================================================

    #[test]
    fn test_gguf_header_write_to() {
        let header = GgufHeader {
            version: 3,
            tensor_count: 10,
            metadata_kv_count: 5,
        };
        let mut buf = Vec::new();
        header.write_to(&mut buf).expect("write header");

        // Check magic (4 bytes)
        assert_eq!(&buf[0..4], &GGUF_MAGIC.to_le_bytes());
        // Check version (4 bytes)
        assert_eq!(&buf[4..8], &3u32.to_le_bytes());
        // Check tensor_count (8 bytes)
        assert_eq!(&buf[8..16], &10u64.to_le_bytes());
        // Check metadata_kv_count (8 bytes)
        assert_eq!(&buf[16..24], &5u64.to_le_bytes());
    }

    #[test]
    fn test_gguf_header_clone_debug() {
        let header = GgufHeader {
            version: 3,
            tensor_count: 1,
            metadata_kv_count: 2,
        };
        let cloned = header.clone();
        assert_eq!(header.version, cloned.version);
        let s = format!("{header:?}");
        assert!(s.contains("GgufHeader"));
    }

    // ============================================================================
    // GgufTensorInfo Tests
    // ============================================================================

    #[test]
    fn test_gguf_tensor_info_write_to() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            n_dims: 2,
            dims: vec![3, 4],
            dtype: GgmlType::F32,
            offset: 128,
        };
        let mut buf = Vec::new();
        info.write_to(&mut buf).expect("write tensor info");

        // Name is length-prefixed: 8 bytes length + "test" (4 bytes)
        assert_eq!(&buf[0..8], &4u64.to_le_bytes());
        assert_eq!(&buf[8..12], b"test");
        // n_dims (4 bytes)
        assert_eq!(&buf[12..16], &2u32.to_le_bytes());
        // dims[0] (8 bytes)
        assert_eq!(&buf[16..24], &3u64.to_le_bytes());
        // dims[1] (8 bytes)
        assert_eq!(&buf[24..32], &4u64.to_le_bytes());
        // dtype (4 bytes)
        assert_eq!(&buf[32..36], &(GgmlType::F32 as u32).to_le_bytes());
        // offset (8 bytes)
        assert_eq!(&buf[36..44], &128u64.to_le_bytes());
    }

    #[test]
    fn test_gguf_tensor_info_clone_debug() {
        let info = GgufTensorInfo {
            name: "layer.0.weight".to_string(),
            n_dims: 3,
            dims: vec![2, 3, 4],
            dtype: GgmlType::F16,
            offset: 0,
        };
        let cloned = info.clone();
        assert_eq!(info.name, cloned.name);
        let s = format!("{info:?}");
        assert!(s.contains("GgufTensorInfo"));
    }

    // ============================================================================
    // GgufTensor Tests
    // ============================================================================

    #[test]
    fn test_gguf_tensor_byte_size_f32() {
        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: vec![10, 20],
            dtype: GgmlType::F32,
            data: vec![],
        };
        // 10 * 20 = 200 elements * 4 bytes = 800
        assert_eq!(tensor.byte_size(), 800);
    }

    #[test]
    fn test_gguf_tensor_byte_size_f16() {
        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: vec![100],
            dtype: GgmlType::F16,
            data: vec![],
        };
        // 100 elements * 2 bytes = 200
        assert_eq!(tensor.byte_size(), 200);
    }

    #[test]
    fn test_gguf_tensor_byte_size_i8() {
        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: vec![256],
            dtype: GgmlType::I8,
            data: vec![],
        };
        // 256 elements * 1 byte = 256
        assert_eq!(tensor.byte_size(), 256);
    }

    #[test]
    fn test_gguf_tensor_byte_size_i16() {
        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: vec![128],
            dtype: GgmlType::I16,
            data: vec![],
        };
        // 128 elements * 2 bytes = 256
        assert_eq!(tensor.byte_size(), 256);
    }

    #[test]
    fn test_gguf_tensor_byte_size_i32() {
        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: vec![64],
            dtype: GgmlType::I32,
            data: vec![],
        };
        // 64 elements * 4 bytes = 256
        assert_eq!(tensor.byte_size(), 256);
    }

    #[test]
    fn test_gguf_tensor_byte_size_i64_f64() {
        let tensor_i64 = GgufTensor {
            name: "test".to_string(),
            shape: vec![32],
            dtype: GgmlType::I64,
            data: vec![],
        };
        // 32 elements * 8 bytes = 256
        assert_eq!(tensor_i64.byte_size(), 256);

        let tensor_f64 = GgufTensor {
            name: "test".to_string(),
            shape: vec![16],
            dtype: GgmlType::F64,
            data: vec![],
        };
        // 16 elements * 8 bytes = 128
        assert_eq!(tensor_f64.byte_size(), 128);
    }

    #[test]
    fn test_gguf_tensor_byte_size_q4_0() {
        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: vec![64],
            dtype: GgmlType::Q4_0,
            data: vec![],
        };
        // 64 elements / 32 = 2 blocks * 18 bytes = 36
        assert_eq!(tensor.byte_size(), 36);
    }

    #[test]
    fn test_gguf_tensor_byte_size_q4_1() {
        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: vec![96],
            dtype: GgmlType::Q4_1,
            data: vec![],
        };
        // (96 + 31) / 32 = 3 blocks * 18 bytes = 54
        assert_eq!(tensor.byte_size(), 54);
    }

    #[test]
    fn test_gguf_tensor_byte_size_q8_0() {
        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: vec![64],
            dtype: GgmlType::Q8_0,
            data: vec![],
        };
        // 64 elements / 32 = 2 blocks * 34 bytes = 68
        assert_eq!(tensor.byte_size(), 68);
    }

    #[test]
    fn test_gguf_tensor_byte_size_q4k() {
        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: vec![512],
            dtype: GgmlType::Q4K,
            data: vec![],
        };
        // 512 elements / 256 = 2 super-blocks * 144 bytes = 288
        assert_eq!(tensor.byte_size(), 288);
    }

    #[test]
    fn test_gguf_tensor_byte_size_q6k() {
        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: vec![256],
            dtype: GgmlType::Q6K,
            data: vec![],
        };
        // 256 elements / 256 = 1 super-block * 210 bytes = 210
        assert_eq!(tensor.byte_size(), 210);
    }

    #[test]
    fn test_gguf_tensor_clone_debug() {
        let tensor = GgufTensor {
            name: "weight".to_string(),
            shape: vec![4, 4],
            dtype: GgmlType::F32,
            data: vec![0u8; 64],
        };
        let cloned = tensor.clone();
        assert_eq!(tensor.name, cloned.name);
        let s = format!("{tensor:?}");
        assert!(s.contains("GgufTensor"));
    }

    // ============================================================================
    // padding_for_alignment Tests
    // ============================================================================

    #[test]
    fn test_padding_for_alignment_already_aligned() {
        assert_eq!(padding_for_alignment(0, 32), 0);
        assert_eq!(padding_for_alignment(32, 32), 0);
        assert_eq!(padding_for_alignment(64, 32), 0);
    }

    #[test]
    fn test_padding_for_alignment_needs_padding() {
        assert_eq!(padding_for_alignment(1, 32), 31);
        assert_eq!(padding_for_alignment(16, 32), 16);
        assert_eq!(padding_for_alignment(31, 32), 1);
        assert_eq!(padding_for_alignment(33, 32), 31);
    }

    #[test]
    fn test_padding_for_alignment_different_alignments() {
        assert_eq!(padding_for_alignment(5, 8), 3);
        assert_eq!(padding_for_alignment(7, 16), 9);
        assert_eq!(padding_for_alignment(100, 64), 28);
    }

    // ============================================================================
    // write_metadata_kv Tests
    // ============================================================================

    #[test]
    fn test_write_metadata_kv_uint32() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "count", &GgufValue::Uint32(42)).expect("write kv");

        // Key: length (8) + "count" (5)
        assert_eq!(&buf[0..8], &5u64.to_le_bytes());
        assert_eq!(&buf[8..13], b"count");
        // Value type: Uint32 = 4
        assert_eq!(&buf[13..17], &4u32.to_le_bytes());
        // Value: 42
        assert_eq!(&buf[17..21], &42u32.to_le_bytes());
    }

    #[test]
    fn test_write_metadata_kv_string() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "arch", &GgufValue::String("llama".to_string())).expect("write kv");

        // Key: length (8) + "arch" (4)
        assert_eq!(&buf[0..8], &4u64.to_le_bytes());
        assert_eq!(&buf[8..12], b"arch");
        // Value type: String = 8
        assert_eq!(&buf[12..16], &8u32.to_le_bytes());
        // Value: length (8) + "llama" (5)
        assert_eq!(&buf[16..24], &5u64.to_le_bytes());
        assert_eq!(&buf[24..29], b"llama");
    }

    #[test]
    fn test_write_metadata_kv_bool() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "flag", &GgufValue::Bool(true)).expect("write kv");
        // Should contain type Bool (7) and value 1
        assert!(buf.len() > 12);
    }

    #[test]
    fn test_write_metadata_kv_all_scalar_types() {
        let cases: Vec<(&str, GgufValue)> = vec![
            ("u8", GgufValue::Uint8(255)),
            ("i8", GgufValue::Int8(-128)),
            ("u16", GgufValue::Uint16(65535)),
            ("i16", GgufValue::Int16(-32768)),
            ("u32", GgufValue::Uint32(u32::MAX)),
            ("i32", GgufValue::Int32(i32::MIN)),
            ("f32", GgufValue::Float32(1.5)),
            ("u64", GgufValue::Uint64(u64::MAX)),
            ("i64", GgufValue::Int64(i64::MIN)),
            ("f64", GgufValue::Float64(2.5)),
        ];

        for (key, value) in cases {
            let mut buf = Vec::new();
            write_metadata_kv(&mut buf, key, &value).expect("write kv");
            assert!(!buf.is_empty(), "Buffer should not be empty for {key}");
        }
    }

    #[test]
    fn test_write_metadata_kv_arrays() {
        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "ids", &GgufValue::ArrayUint32(vec![1, 2, 3])).expect("write kv");
        assert!(!buf.is_empty());

        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "vals", &GgufValue::ArrayInt32(vec![-1, 0, 1])).expect("write kv");
        assert!(!buf.is_empty());

        let mut buf = Vec::new();
        write_metadata_kv(&mut buf, "floats", &GgufValue::ArrayFloat32(vec![1.0, 2.0])).expect("write kv");
        assert!(!buf.is_empty());

        let mut buf = Vec::new();
        write_metadata_kv(
            &mut buf,
            "tokens",
            &GgufValue::ArrayString(vec!["a".to_string(), "b".to_string()]),
        )
        .expect("write kv");
        assert!(!buf.is_empty());
    }

    // ============================================================================
    // export_tensors_to_gguf Tests
    // ============================================================================

    #[test]
    fn test_export_tensors_to_gguf_empty() {
        let mut buf = Vec::new();
        export_tensors_to_gguf(&mut buf, &[], &[]).expect("export empty");

        // Should have header at minimum
        assert!(buf.len() >= 24);
        // Check magic
        assert_eq!(&buf[0..4], &GGUF_MAGIC.to_le_bytes());
    }

    #[test]
    fn test_export_tensors_to_gguf_with_metadata() {
        let metadata = vec![
            ("arch".to_string(), GgufValue::String("test".to_string())),
            ("layers".to_string(), GgufValue::Uint32(4)),
        ];
        let mut buf = Vec::new();
        export_tensors_to_gguf(&mut buf, &[], &metadata).expect("export with metadata");

        // Should have magic
        assert_eq!(&buf[0..4], &GGUF_MAGIC.to_le_bytes());
        // Check metadata count
        assert_eq!(&buf[16..24], &2u64.to_le_bytes());
    }

    #[test]
    fn test_export_tensors_to_gguf_with_tensors() {
        let tensors = vec![
            GgufTensor {
                name: "weight".to_string(),
                shape: vec![4, 4],
                dtype: GgmlType::F32,
                data: vec![0u8; 64], // 4*4*4 bytes
            },
            GgufTensor {
                name: "bias".to_string(),
                shape: vec![4],
                dtype: GgmlType::F32,
                data: vec![0u8; 16], // 4*4 bytes
            },
        ];
        let mut buf = Vec::new();
        export_tensors_to_gguf(&mut buf, &tensors, &[]).expect("export with tensors");

        // Check magic
        assert_eq!(&buf[0..4], &GGUF_MAGIC.to_le_bytes());
        // Check tensor count
        assert_eq!(&buf[8..16], &2u64.to_le_bytes());
    }

    #[test]
    fn test_export_tensors_to_gguf_full() {
        let metadata = vec![
            (
                "general.architecture".to_string(),
                GgufValue::String("llama".to_string()),
            ),
            ("llama.context_length".to_string(), GgufValue::Uint32(2048)),
        ];
        let tensors = vec![GgufTensor {
            name: "model.embed_tokens.weight".to_string(),
            shape: vec![32, 128],
            dtype: GgmlType::F32,
            data: vec![0u8; 32 * 128 * 4],
        }];
        let mut buf = Vec::new();
        export_tensors_to_gguf(&mut buf, &tensors, &metadata).expect("export full");

        // Verify output is properly aligned and contains all data
        assert!(buf.len() > 32 * 128 * 4); // Must be larger than just tensor data
        assert_eq!(&buf[0..4], &GGUF_MAGIC.to_le_bytes());
    }
}
