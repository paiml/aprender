//! GGUF file reader — loads tensor data for inference.
//!
//! Reads GGUF v3 files (llama.cpp compatible). Parses header, metadata,
//! tensor info, then memory-maps or reads tensor data bytes.
//!
//! # Format
//!
//! ```text
//! [magic: u32] [version: u32] [tensor_count: u64] [metadata_kv_count: u64]
//! [metadata KV pairs...]
//! [tensor info entries...]
//! [alignment padding]
//! [tensor data (contiguous)]
//! ```

use std::collections::HashMap;
use std::io::{self, Read, Seek};
use std::path::Path;

use crate::error::TruenoError;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" in little-endian

/// GGML tensor type IDs (subset used for LLM inference).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Bf16 = 30,
}

impl GgmlType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            30 => Some(Self::Bf16),
            _ => None,
        }
    }

    /// Bytes per block for this quantization type.
    pub fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Q4_0 => 18, // 32 weights: 2 (scale) + 16 (4-bit)
            Self::Q4_1 => 20, // 32 weights: 2+2 (scale+min) + 16
            Self::Q5_0 => 22, // 32 weights
            Self::Q5_1 => 24,
            Self::Q8_0 => 34, // 32 weights: 2 (scale) + 32 (8-bit)
            Self::Q8_1 => 36,
            Self::Q2K => 84, // 256 weights
            Self::Q3K => 110,
            Self::Q4K => 144, // 256 weights
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
        }
    }

    /// Weights per block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
        }
    }

    /// Total bytes for `n_elements` weights.
    pub fn tensor_bytes(&self, n_elements: usize) -> usize {
        let bs = self.block_size();
        let n_blocks = (n_elements + bs - 1) / bs;
        n_blocks * self.block_bytes()
    }
}

/// Info about a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: GgmlType,
    pub dims: Vec<u64>,
    /// Offset from start of data section (NOT from file start).
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product::<u64>().max(1)
    }

    pub fn byte_size(&self) -> usize {
        self.dtype.tensor_bytes(self.n_elements() as usize)
    }
}

/// Parsed GGUF file ready for tensor extraction.
pub struct GgufFile {
    pub tensor_count: u64,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    /// Offset in bytes from file start where tensor data begins.
    pub data_offset: u64,
    /// Raw file bytes (memory mapped or loaded).
    data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
}

impl MetadataValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::U64(v) => Some(*v as u32),
            Self::I32(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
}

impl GgufFile {
    /// Load and parse a GGUF file.
    pub fn load(path: &Path) -> Result<Self, TruenoError> {
        let data = std::fs::read(path).map_err(|e| {
            TruenoError::InvalidInput(format!("Failed to read GGUF file {}: {e}", path.display()))
        })?;

        Self::parse(data)
    }

    /// Parse GGUF from raw bytes.
    pub fn parse(data: Vec<u8>) -> Result<Self, TruenoError> {
        let mut cursor = io::Cursor::new(&data);

        // Header
        let magic = read_u32(&mut cursor)?;
        if magic != GGUF_MAGIC {
            return Err(TruenoError::InvalidInput(format!(
                "Not a GGUF file: magic=0x{magic:08x}, expected 0x{GGUF_MAGIC:08x}"
            )));
        }
        let version = read_u32(&mut cursor)?;
        if version < 2 || version > 3 {
            return Err(TruenoError::InvalidInput(format!(
                "Unsupported GGUF version {version} (need 2 or 3)"
            )));
        }
        let tensor_count = read_u64(&mut cursor)?;
        let metadata_kv_count = read_u64(&mut cursor)?;

        // Metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(&mut cursor)?;
            let value = read_metadata_value(&mut cursor)?;
            metadata.insert(key, value);
        }

        // Tensor info
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut cursor)?;
            let n_dims = read_u32(&mut cursor)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut cursor)?);
            }
            let dtype_u32 = read_u32(&mut cursor)?;
            let dtype = GgmlType::from_u32(dtype_u32).ok_or_else(|| {
                TruenoError::InvalidInput(format!(
                    "Unknown GGML type {dtype_u32} for tensor '{name}'"
                ))
            })?;
            let offset = read_u64(&mut cursor)?;
            tensors.push(TensorInfo { name, dtype, dims, offset });
        }

        // Data section starts at next alignment boundary (default 32 bytes)
        let alignment =
            metadata.get("general.alignment").and_then(|v| v.as_u32()).unwrap_or(32) as u64;
        let pos = cursor.position();
        let data_offset = (pos + alignment - 1) / alignment * alignment;

        Ok(Self { tensor_count, metadata, tensors, data_offset, data })
    }

    /// Get raw bytes for a tensor by name.
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensors.iter().find(|t| t.name == name)?;
        let start = self.data_offset as usize + info.offset as usize;
        let end = start + info.byte_size();
        if end <= self.data.len() {
            Some(&self.data[start..end])
        } else {
            None
        }
    }

    /// Get tensor info by name.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get a metadata string value.
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key)?.as_str()
    }

    /// Get a metadata u32 value.
    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key)?.as_u32()
    }

    /// Get a metadata f32 value.
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key)?.as_f32()
    }
}

// ── Binary readers ──

fn read_u8<R: Read>(r: &mut R) -> Result<u8, TruenoError> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)
        .map_err(|e| TruenoError::InvalidInput(format!("GGUF read error: {e}")))?;
    Ok(buf[0])
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16, TruenoError> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)
        .map_err(|e| TruenoError::InvalidInput(format!("GGUF read error: {e}")))?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, TruenoError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| TruenoError::InvalidInput(format!("GGUF read error: {e}")))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(r: &mut R) -> Result<i32, TruenoError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| TruenoError::InvalidInput(format!("GGUF read error: {e}")))?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, TruenoError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)
        .map_err(|e| TruenoError::InvalidInput(format!("GGUF read error: {e}")))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64<R: Read>(r: &mut R) -> Result<i64, TruenoError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)
        .map_err(|e| TruenoError::InvalidInput(format!("GGUF read error: {e}")))?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32_val<R: Read>(r: &mut R) -> Result<f32, TruenoError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| TruenoError::InvalidInput(format!("GGUF read error: {e}")))?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64_val<R: Read>(r: &mut R) -> Result<f64, TruenoError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)
        .map_err(|e| TruenoError::InvalidInput(format!("GGUF read error: {e}")))?;
    Ok(f64::from_le_bytes(buf))
}

fn read_gguf_string<R: Read>(r: &mut R) -> Result<String, TruenoError> {
    let len = read_u64(r)? as usize;
    if len > 1_000_000 {
        return Err(TruenoError::InvalidInput(format!("GGUF string too long: {len}")));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)
        .map_err(|e| TruenoError::InvalidInput(format!("GGUF string read error: {e}")))?;
    String::from_utf8(buf)
        .map_err(|e| TruenoError::InvalidInput(format!("GGUF string not UTF-8: {e}")))
}

fn read_metadata_value<R: Read + Seek>(r: &mut R) -> Result<MetadataValue, TruenoError> {
    let value_type = read_u32(r)?;
    match value_type {
        0 => Ok(MetadataValue::U8(read_u8(r)?)),
        1 => Ok(MetadataValue::I8(read_u8(r)? as i8)),
        2 => Ok(MetadataValue::U16(read_u16(r)?)),
        3 => Ok(MetadataValue::I16(read_u16(r)? as i16)),
        4 => Ok(MetadataValue::U32(read_u32(r)?)),
        5 => Ok(MetadataValue::I32(read_i32(r)?)),
        6 => Ok(MetadataValue::F32(read_f32_val(r)?)),
        7 => Ok(MetadataValue::Bool(read_u8(r)? != 0)),
        8 => Ok(MetadataValue::String(read_gguf_string(r)?)),
        9 => {
            // Array
            let elem_type = read_u32(r)?;
            let count = read_u64(r)? as usize;
            if count > 10_000_000 {
                return Err(TruenoError::InvalidInput(format!("GGUF array too large: {count}")));
            }
            let mut items = Vec::with_capacity(count.min(1024));
            for _ in 0..count {
                // Read elements of the declared type
                let item = match elem_type {
                    0 => MetadataValue::U8(read_u8(r)?),
                    1 => MetadataValue::I8(read_u8(r)? as i8),
                    4 => MetadataValue::U32(read_u32(r)?),
                    5 => MetadataValue::I32(read_i32(r)?),
                    6 => MetadataValue::F32(read_f32_val(r)?),
                    8 => MetadataValue::String(read_gguf_string(r)?),
                    10 => MetadataValue::U64(read_u64(r)?),
                    11 => MetadataValue::I64(read_i64(r)?),
                    12 => MetadataValue::F64(read_f64_val(r)?),
                    _ => {
                        return Err(TruenoError::InvalidInput(format!(
                            "Unsupported GGUF array element type {elem_type}"
                        )))
                    }
                };
                items.push(item);
            }
            Ok(MetadataValue::Array(items))
        }
        10 => Ok(MetadataValue::U64(read_u64(r)?)),
        11 => Ok(MetadataValue::I64(read_i64(r)?)),
        12 => Ok(MetadataValue::F64(read_f64_val(r)?)),
        _ => Err(TruenoError::InvalidInput(format!("Unknown GGUF metadata type {value_type}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_q4k_properties() {
        let q4k = GgmlType::Q4K;
        assert_eq!(q4k.block_size(), 256);
        assert_eq!(q4k.block_bytes(), 144);
        // 4096 weights = 16 blocks × 144 bytes = 2304
        assert_eq!(q4k.tensor_bytes(4096), 2304);
    }

    #[test]
    fn test_ggml_type_f32_properties() {
        let f32t = GgmlType::F32;
        assert_eq!(f32t.block_size(), 1);
        assert_eq!(f32t.block_bytes(), 4);
        assert_eq!(f32t.tensor_bytes(1024), 4096);
    }

    #[test]
    fn test_gguf_magic_check() {
        let bad_data = vec![0u8; 32];
        let result = GgufFile::parse(bad_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_minimal_gguf() {
        // Build a minimal valid GGUF v3 with 0 tensors, 0 metadata
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes()); // magic
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count
                                                     // Pad to 32-byte alignment
        data.resize(32, 0);

        let file = GgufFile::parse(data).expect("valid minimal GGUF");
        assert_eq!(file.tensor_count, 0);
        assert_eq!(file.tensors.len(), 0);
    }
}
