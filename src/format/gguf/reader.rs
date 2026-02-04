//! GGUF reader and binary parsing (spec §7.2)

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

// BUG-GGUF-001 FIX: Define reasonable limits to prevent allocation attacks
// A malicious file with tensor_count=u64::MAX could cause OOM or panic.
// Even the largest models (Llama 405B) have <1000 tensors.
/// Maximum number of tensors allowed in a GGUF file (prevents OOM attack)
const MAX_TENSOR_COUNT: u64 = 100_000;
/// Maximum number of metadata entries allowed (prevents OOM attack)
const MAX_METADATA_COUNT: u64 = 100_000;
/// Maximum number of dimensions per tensor (no real tensor has > 8 dims)
const MAX_DIMS: u32 = 16;

use super::dequant::{
    dequantize_iq_approximate, dequantize_q2_k, dequantize_q3_k, dequantize_q4_k, dequantize_q5_1,
    dequantize_q5_k, dequantize_q6_k, f16_to_f32,
};
use super::types::{
    padding_for_alignment, GgufValue, TensorDataMap, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC,
};
use crate::error::{AprenderError, Result};

// ============================================================================
// GGUF Reading/Import API
// ============================================================================

/// Read a u32 from bytes at offset
pub(crate) fn read_u32(data: &[u8], offset: usize) -> Result<u32> {
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
pub(crate) fn read_u64(data: &[u8], offset: usize) -> Result<u64> {
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
pub(crate) fn read_string(data: &[u8], offset: usize) -> Result<(String, usize)> {
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
pub(crate) fn read_metadata_value(
    data: &[u8],
    offset: usize,
    value_type: u32,
) -> Result<(GgufValue, usize)> {
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
    pub(crate) data: Vec<u8>,
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

        // Check magic (GH-183: Enhanced error message for debugging)
        let magic = read_u32(&data, 0)?;
        if magic != GGUF_MAGIC {
            // Show both hex and ASCII for easier debugging
            let magic_bytes = &data[0..4.min(data.len())];
            let magic_ascii: String = magic_bytes
                .iter()
                .map(|&b| if b.is_ascii_graphic() { b as char } else { '.' })
                .collect();
            return Err(AprenderError::FormatError {
                message: format!(
                    "Invalid GGUF magic: 0x{magic:08X} (bytes: {magic_bytes:02X?}, ascii: \"{magic_ascii}\"), \
                     expected 0x{GGUF_MAGIC:08X} (\"GGUF\")"
                ),
            });
        }

        let version = read_u32(&data, 4)?;
        let tensor_count = read_u64(&data, 8)?;
        let metadata_kv_count = read_u64(&data, 16)?;

        // BUG-GGUF-001 FIX: Validate counts before allocation to prevent OOM attacks
        if tensor_count > MAX_TENSOR_COUNT {
            return Err(AprenderError::FormatError {
                message: format!(
                    "GGUF tensor_count {} exceeds maximum allowed {} (possible corrupted/malicious file)",
                    tensor_count, MAX_TENSOR_COUNT
                ),
            });
        }
        if metadata_kv_count > MAX_METADATA_COUNT {
            return Err(AprenderError::FormatError {
                message: format!(
                    "GGUF metadata_kv_count {} exceeds maximum allowed {} (possible corrupted/malicious file)",
                    metadata_kv_count, MAX_METADATA_COUNT
                ),
            });
        }

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
            let n_dims = read_u32(&data, offset)?;
            offset += 4;

            // BUG-GGUF-001 FIX: Validate n_dims to prevent allocation attacks
            if n_dims > MAX_DIMS {
                return Err(AprenderError::FormatError {
                    message: format!(
                        "Tensor '{}' has {} dimensions, exceeds maximum {} (possible corrupted file)",
                        name, n_dims, MAX_DIMS
                    ),
                });
            }

            // Read dimensions
            let mut dims = Vec::with_capacity(n_dims as usize);
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

    /// Get BPE merge rules from metadata (PMAT-171)
    ///
    /// Returns the merge rules as "token1 token2" strings for BPE encoding.
    /// Uses "tokenizer.ggml.merges" key from GGUF metadata.
    #[must_use]
    pub fn merges(&self) -> Option<Vec<String>> {
        if let Some(GgufValue::ArrayString(merges)) = self.metadata.get("tokenizer.ggml.merges") {
            if merges.is_empty() {
                None
            } else {
                Some(merges.clone())
            }
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
            // GGML dtype values (from ggml.h):
            // 0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 6=Q5_0, 7=Q5_1, 8=Q8_0, 9=Q8_1
            // 10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K
            // 16+=IQ variants
            2 => {
                // Q4_0 - dequantize
                super::dequantize_q4_0(&self.data, tensor_start, num_elements)?
            }
            3 => {
                // Q4_1 - dequantize (blocks of 32 with scale and min)
                super::dequantize_q4_1(&self.data, tensor_start, num_elements)?
            }
            6 => {
                // Q5_0 - dequantize (blocks of 32 with 5-bit quants)
                super::dequantize_q5_0(&self.data, tensor_start, num_elements)?
            }
            7 => {
                // Q5_1 - dequantize (blocks of 32 with 5-bit quants + min)
                dequantize_q5_1(&self.data, tensor_start, num_elements)?
            }
            8 => {
                // Q8_0 - dequantize
                super::dequantize_q8_0(&self.data, tensor_start, num_elements)?
            }
            10 => {
                // Q2_K - dequantize (super blocks of 256)
                dequantize_q2_k(&self.data, tensor_start, num_elements)?
            }
            11 => {
                // Q3_K - dequantize (super blocks of 256)
                dequantize_q3_k(&self.data, tensor_start, num_elements)?
            }
            12 => {
                // Q4_K - dequantize (super blocks of 256 elements, 144 bytes/block)
                dequantize_q4_k(&self.data, tensor_start, num_elements)?
            }
            13 => {
                // Q5_K - dequantize (super blocks of 256 elements, 176 bytes/block)
                dequantize_q5_k(&self.data, tensor_start, num_elements)?
            }
            14 => {
                // Q6_K - dequantize (super blocks of 256 elements, 210 bytes/block)
                dequantize_q6_k(&self.data, tensor_start, num_elements)?
            }
            // I-quants (importance matrix quantization) - complex formats
            // For now, approximate with simpler dequantization
            16..=23 => {
                // IQ2_XXS=16, IQ2_XS=17, IQ2_S=18, IQ3_XXS=19, IQ3_S=20, IQ1_S=21, IQ4_NL=22, IQ4_XS=23
                // Fall back to zero-filled tensor with warning
                eprintln!(
                    "Warning: I-quant dtype {} for tensor '{}' not fully supported, using approximation",
                    meta.dtype, name
                );
                dequantize_iq_approximate(&self.data, tensor_start, num_elements, meta.dtype)
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
    /// Returns (raw_bytes, shape, ggml_dtype) where dtype is per GGML spec:
    /// - 0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 8=Q8_0
    /// - 10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K
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

        // Calculate byte size based on dtype (GGML dtype values)
        // See llama.cpp ggml.h for type definitions
        // GGML enum: 0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 6=Q5_0, 7=Q5_1, 8=Q8_0, 9=Q8_1
        //           10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K, 15=Q8_K
        // Note: Some dtypes share byte sizes but are documented separately for clarity
        #[allow(clippy::match_same_arms)]
        let byte_size = match meta.dtype {
            0 => num_elements * 4,         // F32
            1 => num_elements * 2,         // F16
            2 => (num_elements / 32) * 18, // Q4_0: 32 elements = 2 (d) + 16 (qs)
            3 => (num_elements / 32) * 20, // Q4_1: 32 elements = 2 (d) + 2 (m) + 16 (qs)
            // dtype 4,5 = removed (Q4_2, Q4_3)
            6 => (num_elements / 32) * 22, // Q5_0: 32 elements = 2 (d) + 4 (qh) + 16 (ql)
            7 => (num_elements / 32) * 24, // Q5_1: 32 elements = 2 (d) + 2 (m) + 4 (qh) + 16 (ql)
            8 => (num_elements / 32) * 34, // Q8_0: 32 elements = 2 (d) + 32 (qs)
            9 => (num_elements / 32) * 36, // Q8_1: 32 elements = 2 (d) + 2 (s) + 32 (qs)
            10 => (num_elements / 256) * 84, // Q2_K: 256 elements = 84 bytes
            11 => (num_elements / 256) * 110, // Q3_K: 256 elements = 110 bytes
            12 => (num_elements / 256) * 144, // Q4_K: 256 elements = 144 bytes
            13 => (num_elements / 256) * 176, // Q5_K: 256 elements = 176 bytes
            14 => (num_elements / 256) * 210, // Q6_K: 256 elements = 210 bytes
            15 => (num_elements / 256) * 292, // Q8_K: 256 elements = 292 bytes
            30 => num_elements * 2,        // BF16: 2 bytes per element
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

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // BUG-GGUF-001 Falsification Tests: Allocation Attack Prevention
    // ========================================================================

    /// Create minimal GGUF header bytes for testing
    fn create_gguf_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
        let mut data = Vec::new();
        // Magic: "GGUF"
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version: 3
        data.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count
        data.extend_from_slice(&tensor_count.to_le_bytes());
        // Metadata count
        data.extend_from_slice(&metadata_count.to_le_bytes());
        data
    }

    #[test]
    fn test_bug_gguf_001_excessive_tensor_count_rejected() {
        // Create GGUF with tensor_count > MAX_TENSOR_COUNT
        let data = create_gguf_header(MAX_TENSOR_COUNT + 1, 0);

        let result = GgufReader::from_bytes(data);
        assert!(
            result.is_err(),
            "FALSIFIED: Excessive tensor_count should be rejected"
        );
        let err = format!("{:?}", result.unwrap_err());
        assert!(
            err.contains("exceeds maximum"),
            "Error should mention limit: {err}"
        );
    }

    #[test]
    fn test_bug_gguf_001_excessive_metadata_count_rejected() {
        // Create GGUF with metadata_kv_count > MAX_METADATA_COUNT
        let data = create_gguf_header(1, MAX_METADATA_COUNT + 1);

        let result = GgufReader::from_bytes(data);
        assert!(
            result.is_err(),
            "FALSIFIED: Excessive metadata_kv_count should be rejected"
        );
        let err = format!("{:?}", result.unwrap_err());
        assert!(
            err.contains("exceeds maximum"),
            "Error should mention limit: {err}"
        );
    }

    #[test]
    fn test_bug_gguf_001_max_tensor_count_allowed() {
        // Create GGUF with tensor_count = MAX_TENSOR_COUNT (should be allowed)
        // Will fail due to truncated file, but NOT due to count validation
        let data = create_gguf_header(MAX_TENSOR_COUNT, 0);

        let result = GgufReader::from_bytes(data);
        // Will fail because file is truncated, but NOT because of tensor_count
        match result {
            Err(e) => {
                let err = format!("{e:?}");
                assert!(
                    !err.contains("tensor_count") || !err.contains("exceeds"),
                    "MAX_TENSOR_COUNT should be accepted: {err}"
                );
            }
            Ok(_) => {
                // Unlikely but acceptable
            }
        }
    }

    #[test]
    fn test_bug_gguf_001_zero_counts_valid() {
        // Zero tensor/metadata counts are valid (empty model)
        let data = create_gguf_header(0, 0);

        // Will succeed or fail due to other reasons (no tensor data), not counts
        let result = GgufReader::from_bytes(data);
        match result {
            Err(e) => {
                let err = format!("{e:?}");
                assert!(
                    !err.contains("exceeds maximum"),
                    "Zero counts should be valid: {err}"
                );
            }
            Ok(_) => {
                // Valid: empty GGUF file
            }
        }
    }
}
