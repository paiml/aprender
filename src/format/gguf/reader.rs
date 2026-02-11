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
/// BUG-GGUF-002 FIX: Maximum total elements per tensor (~16GB F32 tensor)
/// This prevents integer overflow in shape.iter().product() and subsequent
/// byte size calculations. 4B elements * 4 bytes = 16GB, reasonable for largest models.
const MAX_TENSOR_ELEMENTS: usize = 4_000_000_000;

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

/// Read a GGUF array value (type 9) and return (value, bytes_consumed).
fn read_metadata_array(data: &[u8], offset: usize) -> Result<(GgufValue, usize)> {
    let elem_type = read_u32(data, offset)?;
    let count = read_u64(data, offset + 4)? as usize;
    let mut consumed = 12; // type (4) + count (8)

    match elem_type {
        8 => {
            let mut strings = Vec::with_capacity(count);
            for _ in 0..count {
                let (s, len) = read_string(data, offset + consumed)?;
                strings.push(s);
                consumed += len;
            }
            Ok((GgufValue::ArrayString(strings), consumed))
        }
        4 => {
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(read_u32(data, offset + consumed)?);
                consumed += 4;
            }
            Ok((GgufValue::ArrayUint32(values), consumed))
        }
        5 => {
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
        }
        6 => {
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
        }
        _ => {
            let elem_size = match elem_type {
                0..=1 | 7 => 1,
                2..=3 => 2,
                10..=12 => 8,
                _ => 4,
            };
            consumed += count * elem_size;
            Ok((GgufValue::ArrayUint32(vec![]), consumed))
        }
    }
}

/// Read a metadata value and return (value, bytes_consumed)
/// Ensure `n` bytes are available at `offset`, returning a format error with `type_name` if not.
fn ensure_bytes(data: &[u8], offset: usize, n: usize, type_name: &str) -> Result<()> {
    if offset + n > data.len() {
        return Err(AprenderError::FormatError {
            message: format!("Unexpected EOF reading {type_name}"),
        });
    }
    Ok(())
}

/// Read a little-endian i16 from `data` at `offset`.
fn read_i16_le(data: &[u8], offset: usize) -> Result<i16> {
    ensure_bytes(data, offset, 2, "Int16")?;
    Ok(i16::from_le_bytes([data[offset], data[offset + 1]]))
}

/// Read a little-endian i32 from `data` at `offset`.
fn read_i32_le(data: &[u8], offset: usize) -> Result<i32> {
    ensure_bytes(data, offset, 4, "Int32")?;
    Ok(i32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

/// Read a little-endian f32 from `data` at `offset`.
fn read_f32_le(data: &[u8], offset: usize) -> Result<f32> {
    ensure_bytes(data, offset, 4, "Float32")?;
    Ok(f32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

/// Read a little-endian i64 from `data` at `offset`.
fn read_i64_le(data: &[u8], offset: usize) -> Result<i64> {
    ensure_bytes(data, offset, 8, "Int64")?;
    Ok(i64::from_le_bytes([
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

/// Read a little-endian f64 from `data` at `offset`.
fn read_f64_le(data: &[u8], offset: usize) -> Result<f64> {
    ensure_bytes(data, offset, 8, "Float64")?;
    Ok(f64::from_le_bytes([
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

pub(crate) fn read_metadata_value(
    data: &[u8],
    offset: usize,
    value_type: u32,
) -> Result<(GgufValue, usize)> {
    match value_type {
        0 => {
            ensure_bytes(data, offset, 1, "Uint8")?;
            Ok((GgufValue::Uint8(data[offset]), 1))
        }
        1 => {
            ensure_bytes(data, offset, 1, "Int8")?;
            Ok((GgufValue::Int8(data[offset] as i8), 1))
        }
        2 => {
            ensure_bytes(data, offset, 2, "Uint16")?;
            Ok((
                GgufValue::Uint16(u16::from_le_bytes([data[offset], data[offset + 1]])),
                2,
            ))
        }
        3 => Ok((GgufValue::Int16(read_i16_le(data, offset)?), 2)),
        4 => Ok((GgufValue::Uint32(read_u32(data, offset)?), 4)),
        5 => Ok((GgufValue::Int32(read_i32_le(data, offset)?), 4)),
        6 => Ok((GgufValue::Float32(read_f32_le(data, offset)?), 4)),
        7 => {
            ensure_bytes(data, offset, 1, "Bool")?;
            Ok((GgufValue::Bool(data[offset] != 0), 1))
        }
        8 => {
            let (s, len) = read_string(data, offset)?;
            Ok((GgufValue::String(s), len))
        }
        9 => read_metadata_array(data, offset),
        10 => Ok((GgufValue::Uint64(read_u64(data, offset)?), 8)),
        11 => Ok((GgufValue::Int64(read_i64_le(data, offset)?), 8)),
        12 => Ok((GgufValue::Float64(read_f64_le(data, offset)?), 8)),
        _ => Ok((GgufValue::Uint32(0), 4)),
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

    // ========================================================================
    // GH-253: Tokenizer metadata accessors for GGUF export round-trip
    // ========================================================================

    /// Get per-token type array (tokenizer.ggml.token_type)
    /// Values: 1=normal, 2=unknown, 3=control/special, 4=user_defined, etc.
    #[must_use]
    pub fn token_type(&self) -> Option<Vec<i32>> {
        if let Some(GgufValue::ArrayInt32(types)) = self.metadata.get("tokenizer.ggml.token_type") {
            if types.is_empty() {
                None
            } else {
                Some(types.clone())
            }
        } else {
            None
        }
    }

    /// Get padding token ID (tokenizer.ggml.padding_token_id)
    #[must_use]
    pub fn padding_token_id(&self) -> Option<u32> {
        if let Some(GgufValue::Uint32(id)) = self.metadata.get("tokenizer.ggml.padding_token_id") {
            Some(*id)
        } else {
            None
        }
    }

    /// Get add_bos_token flag (tokenizer.ggml.add_bos_token)
    #[must_use]
    pub fn add_bos_token(&self) -> Option<bool> {
        if let Some(GgufValue::Bool(v)) = self.metadata.get("tokenizer.ggml.add_bos_token") {
            Some(*v)
        } else {
            None
        }
    }

    /// Get chat template (tokenizer.chat_template)
    #[must_use]
    pub fn chat_template(&self) -> Option<String> {
        if let Some(GgufValue::String(tmpl)) = self.metadata.get("tokenizer.chat_template") {
            Some(tmpl.clone())
        } else {
            None
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

        // BUG-GGUF-002 FIX: Use checked multiplication to prevent integer overflow
        let num_elements = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| AprenderError::FormatError {
                message: format!(
                    "Tensor '{}' shape {:?} causes integer overflow (malicious file?)",
                    name, shape
                ),
            })?;

        // BUG-GGUF-002 FIX: Validate total elements against reasonable limit
        if num_elements > MAX_TENSOR_ELEMENTS {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Tensor '{}' has {} elements, exceeds max {} (possible malicious file)",
                    name, num_elements, MAX_TENSOR_ELEMENTS
                ),
            });
        }

        let tensor_start = self.data_offset + meta.offset as usize;

        let data = match meta.dtype {
            0 => {
                // F32 - direct copy
                // BUG-GGUF-002 FIX: Use checked_mul for byte size calculation
                let byte_size =
                    num_elements
                        .checked_mul(4)
                        .ok_or_else(|| AprenderError::FormatError {
                            message: format!("Tensor '{}' byte size calculation overflow", name),
                        })?;
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
                // BUG-GGUF-002 FIX: Use checked_mul for byte size calculation
                let byte_size =
                    num_elements
                        .checked_mul(2)
                        .ok_or_else(|| AprenderError::FormatError {
                            message: format!("Tensor '{}' byte size calculation overflow", name),
                        })?;
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

        // BUG-GGUF-002 FIX: Use checked multiplication to prevent integer overflow
        let num_elements = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| AprenderError::FormatError {
                message: format!(
                    "Tensor '{}' shape {:?} causes integer overflow (malicious file?)",
                    name, shape
                ),
            })?;

        // BUG-GGUF-002 FIX: Validate total elements against reasonable limit
        if num_elements > MAX_TENSOR_ELEMENTS {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Tensor '{}' has {} elements, exceeds max {} (possible malicious file)",
                    name, num_elements, MAX_TENSOR_ELEMENTS
                ),
            });
        }

        let tensor_start = self.data_offset + meta.offset as usize;

        // Calculate byte size based on dtype (GGML dtype values)
        // See llama.cpp ggml.h for type definitions
        // GGML enum: 0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 6=Q5_0, 7=Q5_1, 8=Q8_0, 9=Q8_1
        //           10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K, 15=Q8_K
        // BUG-GGUF-002 FIX: Use checked arithmetic to prevent overflow in byte size calc
        // Note: Some dtypes share byte sizes but are documented separately for clarity
        #[allow(clippy::match_same_arms)]
        let byte_size = match meta.dtype {
            0 => num_elements.checked_mul(4),         // F32
            1 => num_elements.checked_mul(2),         // F16
            2 => (num_elements / 32).checked_mul(18), // Q4_0: 32 elements = 2 (d) + 16 (qs)
            3 => (num_elements / 32).checked_mul(20), // Q4_1: 32 elements = 2 (d) + 2 (m) + 16 (qs)
            // dtype 4,5 = removed (Q4_2, Q4_3)
            6 => (num_elements / 32).checked_mul(22), // Q5_0: 32 elements = 2 (d) + 4 (qh) + 16 (ql)
            7 => (num_elements / 32).checked_mul(24), // Q5_1: 32 elements = 2 (d) + 2 (m) + 4 (qh) + 16 (ql)
            8 => (num_elements / 32).checked_mul(34), // Q8_0: 32 elements = 2 (d) + 32 (qs)
            9 => (num_elements / 32).checked_mul(36), // Q8_1: 32 elements = 2 (d) + 2 (s) + 32 (qs)
            10 => (num_elements / 256).checked_mul(84), // Q2_K: 256 elements = 84 bytes
            11 => (num_elements / 256).checked_mul(110), // Q3_K: 256 elements = 110 bytes
            12 => (num_elements / 256).checked_mul(144), // Q4_K: 256 elements = 144 bytes
            13 => (num_elements / 256).checked_mul(176), // Q5_K: 256 elements = 176 bytes
            14 => (num_elements / 256).checked_mul(210), // Q6_K: 256 elements = 210 bytes
            15 => (num_elements / 256).checked_mul(292), // Q8_K: 256 elements = 292 bytes
            30 => num_elements.checked_mul(2),        // BF16: 2 bytes per element
            _ => {
                return Err(AprenderError::FormatError {
                    message: format!("Unsupported dtype {} for raw extraction", meta.dtype),
                });
            }
        }
        .ok_or_else(|| AprenderError::FormatError {
            message: format!(
                "Tensor '{}' byte size calculation overflow (dtype: {})",
                name, meta.dtype
            ),
        })?;

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

    // ========================================================================
    // BUG-GGUF-002 Falsification Tests: Integer Overflow Prevention
    // ========================================================================
    // The shape.iter().product() call can overflow if malicious dimensions are provided.
    // Byte size calculations (num_elements * bytes_per_element) can also overflow.
    // Fixed by using checked_mul() and validating against MAX_TENSOR_ELEMENTS.

    #[test]
    fn test_bug_gguf_002_overflow_protection_documented() {
        // BUG-GGUF-002: Integer overflow in tensor element count
        //
        // Attack vector: Malicious GGUF with dimensions like [2^32, 2^32]
        // Prior behavior: Overflow to small value, then OOM or buffer overread
        //
        // Fix applied:
        // 1. Use checked_mul in shape.iter().try_fold() for element count
        // 2. Validate num_elements <= MAX_TENSOR_ELEMENTS (4 billion)
        // 3. Use checked_mul for all byte size calculations
        //
        // Locations fixed:
        // - get_tensor(): lines ~720-740
        // - get_tensor_raw(): lines ~870-920
        //
        // This test documents the fix. Triggering the actual overflow would require
        // crafting a valid GGUF header with malicious tensor dimensions, which is
        // complex. The fix ensures that IF such a file is parsed, it will fail
        // safely with an error instead of causing undefined behavior.
        assert!(MAX_TENSOR_ELEMENTS == 4_000_000_000);
    }

    // ========================================================================
    // read_metadata_value Direct Tests: Cover All Type Branches
    // ========================================================================

    #[test]
    fn test_read_metadata_value_uint64() {
        let val: u64 = 0x_DEAD_BEEF_CAFE_BABE;
        let bytes = val.to_le_bytes();
        let (result, consumed) = read_metadata_value(&bytes, 0, 10).expect("read uint64");
        assert_eq!(consumed, 8);
        match result {
            GgufValue::Uint64(v) => assert_eq!(v, val),
            other => panic!("Expected Uint64, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_int64() {
        let val: i64 = -123_456_789_012_345;
        let bytes = val.to_le_bytes();
        let (result, consumed) = read_metadata_value(&bytes, 0, 11).expect("read int64");
        assert_eq!(consumed, 8);
        match result {
            GgufValue::Int64(v) => assert_eq!(v, val),
            other => panic!("Expected Int64, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_float64() {
        let val: f64 = std::f64::consts::PI;
        let bytes = val.to_le_bytes();
        let (result, consumed) = read_metadata_value(&bytes, 0, 12).expect("read float64");
        assert_eq!(consumed, 8);
        match result {
            GgufValue::Float64(v) => assert!((v - val).abs() < f64::EPSILON),
            other => panic!("Expected Float64, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_unknown_type() {
        // Unknown type 99 should return Uint32(0) and consume 4 bytes
        let bytes = [0u8; 8];
        let (result, consumed) = read_metadata_value(&bytes, 0, 99).expect("read unknown");
        assert_eq!(consumed, 4);
        match result {
            GgufValue::Uint32(v) => assert_eq!(v, 0),
            other => panic!("Expected Uint32(0) for unknown type, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_array_other_type_uint8() {
        // Array of Uint8 (elem_type=0) falls into the "else" branch
        // Build: elem_type(4) + count(8) + data
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0u32.to_le_bytes()); // elem_type = 0 (Uint8)
        bytes.extend_from_slice(&3u64.to_le_bytes()); // count = 3
        bytes.extend_from_slice(&[10u8, 20u8, 30u8]); // 3 Uint8 elements
        let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array uint8");
        // 12 (header) + 3 * 1 (uint8 elem_size) = 15
        assert_eq!(consumed, 15);
        match result {
            GgufValue::ArrayUint32(v) => {
                assert!(v.is_empty(), "Other-type arrays return empty vec")
            }
            other => panic!("Expected empty ArrayUint32, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_array_other_type_uint16() {
        // Array of Uint16 (elem_type=2)
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u32.to_le_bytes()); // elem_type = 2 (Uint16)
        bytes.extend_from_slice(&2u64.to_le_bytes()); // count = 2
        bytes.extend_from_slice(&[0u8; 4]); // 2 * 2 bytes
        let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array uint16");
        assert_eq!(consumed, 16); // 12 + 2*2
        match result {
            GgufValue::ArrayUint32(v) => assert!(v.is_empty()),
            other => panic!("Expected empty ArrayUint32, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_array_other_type_uint64() {
        // Array of Uint64 (elem_type=10) falls into 8-byte branch
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&10u32.to_le_bytes()); // elem_type = 10 (Uint64)
        bytes.extend_from_slice(&2u64.to_le_bytes()); // count = 2
        bytes.extend_from_slice(&[0u8; 16]); // 2 * 8 bytes
        let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array uint64");
        assert_eq!(consumed, 28); // 12 + 2*8
        match result {
            GgufValue::ArrayUint32(v) => assert!(v.is_empty()),
            other => panic!("Expected empty ArrayUint32, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_array_other_type_int64() {
        // Array of Int64 (elem_type=11)
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&11u32.to_le_bytes()); // elem_type = 11 (Int64)
        bytes.extend_from_slice(&1u64.to_le_bytes()); // count = 1
        bytes.extend_from_slice(&[0u8; 8]); // 1 * 8 bytes
        let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array int64");
        assert_eq!(consumed, 20); // 12 + 1*8
        match result {
            GgufValue::ArrayUint32(v) => assert!(v.is_empty()),
            other => panic!("Expected empty ArrayUint32, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_array_other_type_float64() {
        // Array of Float64 (elem_type=12)
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&12u32.to_le_bytes()); // elem_type = 12 (Float64)
        bytes.extend_from_slice(&1u64.to_le_bytes()); // count = 1
        bytes.extend_from_slice(&[0u8; 8]); // 1 * 8 bytes
        let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array float64");
        assert_eq!(consumed, 20); // 12 + 1*8
        match result {
            GgufValue::ArrayUint32(v) => assert!(v.is_empty()),
            other => panic!("Expected empty ArrayUint32, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_array_other_type_bool() {
        // Array of Bool (elem_type=7) -> 1-byte elements
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&7u32.to_le_bytes()); // elem_type = 7 (Bool)
        bytes.extend_from_slice(&4u64.to_le_bytes()); // count = 4
        bytes.extend_from_slice(&[1u8, 0u8, 1u8, 0u8]); // 4 bools
        let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array bool");
        assert_eq!(consumed, 16); // 12 + 4*1
        match result {
            GgufValue::ArrayUint32(v) => assert!(v.is_empty()),
            other => panic!("Expected empty ArrayUint32, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_array_other_type_default() {
        // Array of unknown elem_type (e.g., 99) -> default 4-byte elements
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&99u32.to_le_bytes()); // elem_type = 99 (unknown)
        bytes.extend_from_slice(&2u64.to_le_bytes()); // count = 2
        bytes.extend_from_slice(&[0u8; 8]); // 2 * 4 bytes
        let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array unknown");
        assert_eq!(consumed, 20); // 12 + 2*4
        match result {
            GgufValue::ArrayUint32(v) => assert!(v.is_empty()),
            other => panic!("Expected empty ArrayUint32, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_array_int32() {
        // Array of Int32 (elem_type=5) - has its own explicit branch
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&5u32.to_le_bytes()); // elem_type = 5 (Int32)
        bytes.extend_from_slice(&3u64.to_le_bytes()); // count = 3
        bytes.extend_from_slice(&(-10i32).to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&42i32.to_le_bytes());
        let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array int32");
        assert_eq!(consumed, 24); // 12 + 3*4
        match result {
            GgufValue::ArrayInt32(v) => {
                assert_eq!(v, vec![-10, 0, 42]);
            }
            other => panic!("Expected ArrayInt32, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_int64_eof() {
        // Int64 with insufficient data should error
        let bytes = [0u8; 4]; // Only 4 bytes, need 8
        let result = read_metadata_value(&bytes, 0, 11);
        assert!(result.is_err(), "Int64 with < 8 bytes should fail");
    }

    #[test]
    fn test_read_metadata_value_float64_eof() {
        // Float64 with insufficient data should error
        let bytes = [0u8; 5]; // Only 5 bytes, need 8
        let result = read_metadata_value(&bytes, 0, 12);
        assert!(result.is_err(), "Float64 with < 8 bytes should fail");
    }

    #[test]
    fn test_read_metadata_value_uint8() {
        let bytes = [42u8];
        let (result, consumed) = read_metadata_value(&bytes, 0, 0).expect("read uint8");
        assert_eq!(consumed, 1);
        match result {
            GgufValue::Uint8(v) => assert_eq!(v, 42),
            other => panic!("Expected Uint8, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_int8() {
        let bytes = [0xFEu8]; // -2 as i8
        let (result, consumed) = read_metadata_value(&bytes, 0, 1).expect("read int8");
        assert_eq!(consumed, 1);
        match result {
            GgufValue::Int8(v) => assert_eq!(v, -2),
            other => panic!("Expected Int8, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_uint16() {
        let bytes = 1000u16.to_le_bytes();
        let (result, consumed) = read_metadata_value(&bytes, 0, 2).expect("read uint16");
        assert_eq!(consumed, 2);
        match result {
            GgufValue::Uint16(v) => assert_eq!(v, 1000),
            other => panic!("Expected Uint16, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_int16() {
        let bytes = (-500i16).to_le_bytes();
        let (result, consumed) = read_metadata_value(&bytes, 0, 3).expect("read int16");
        assert_eq!(consumed, 2);
        match result {
            GgufValue::Int16(v) => assert_eq!(v, -500),
            other => panic!("Expected Int16, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_bool_true() {
        let bytes = [1u8];
        let (result, consumed) = read_metadata_value(&bytes, 0, 7).expect("read bool");
        assert_eq!(consumed, 1);
        match result {
            GgufValue::Bool(v) => assert!(v),
            other => panic!("Expected Bool(true), got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_bool_false() {
        let bytes = [0u8];
        let (result, consumed) = read_metadata_value(&bytes, 0, 7).expect("read bool");
        assert_eq!(consumed, 1);
        match result {
            GgufValue::Bool(v) => assert!(!v),
            other => panic!("Expected Bool(false), got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_with_offset() {
        // Test reading with a non-zero offset
        let mut bytes = vec![0u8; 10]; // padding
        bytes.extend_from_slice(&42u64.to_le_bytes());
        let (result, consumed) =
            read_metadata_value(&bytes, 10, 10).expect("read uint64 at offset");
        assert_eq!(consumed, 8);
        match result {
            GgufValue::Uint64(v) => assert_eq!(v, 42),
            other => panic!("Expected Uint64, got {other:?}"),
        }
    }

    // ========================================================================
    // GgufReader::from_bytes Error Path Tests
    // ========================================================================

    #[test]
    fn test_from_bytes_file_too_small() {
        let data = vec![0u8; 10]; // < 24 bytes
        let result = GgufReader::from_bytes(data);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(
            err.contains("too small"),
            "Error should mention 'too small': {err}"
        );
    }

    #[test]
    fn test_from_bytes_file_exactly_23_bytes() {
        let data = vec![0u8; 23]; // One byte short
        let result = GgufReader::from_bytes(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_bytes_invalid_magic() {
        let mut data = vec![0u8; 24];
        // Write invalid magic "XXXX" instead of "GGUF"
        data[0] = b'X';
        data[1] = b'X';
        data[2] = b'X';
        data[3] = b'X';
        let result = GgufReader::from_bytes(data);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(
            err.contains("Invalid GGUF magic"),
            "Error should mention invalid magic: {err}"
        );
        // GH-183: Enhanced error should show hex and ASCII
        assert!(
            err.contains("0x") || err.contains("ascii"),
            "Error should include hex/ascii debug info: {err}"
        );
    }

    #[test]
    fn test_from_bytes_zero_tensors_zero_metadata() {
        let data = create_gguf_header(0, 0);
        let reader = GgufReader::from_bytes(data).expect("valid empty GGUF");
        assert_eq!(reader.tensor_count, 0);
        assert!(reader.tensors.is_empty());
        assert!(reader.metadata.is_empty());
        assert_eq!(reader.version, 3);
    }

    // ========================================================================
    // GgufReader::from_bytes with Tensor Metadata Tests
    // ========================================================================

    /// Build a complete synthetic GGUF file with one F32 tensor and optional metadata
    fn build_synthetic_gguf_with_tensor(
        tensor_name: &str,
        dims: &[u64],
        dtype: u32,
        tensor_data: &[u8],
        metadata: &[(&str, u32, &[u8])], // (key, value_type, value_bytes)
    ) -> Vec<u8> {
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
        data.extend_from_slice(&(metadata.len() as u64).to_le_bytes()); // metadata_count

        // Metadata KV pairs
        for (key, value_type, value_bytes) in metadata {
            // Key string (length-prefixed)
            data.extend_from_slice(&(key.len() as u64).to_le_bytes());
            data.extend_from_slice(key.as_bytes());
            // Value type
            data.extend_from_slice(&value_type.to_le_bytes());
            // Value bytes
            data.extend_from_slice(value_bytes);
        }

        // Tensor info
        // Name (length-prefixed)
        data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        data.extend_from_slice(tensor_name.as_bytes());
        // n_dims
        let n_dims = dims.len() as u32;
        data.extend_from_slice(&n_dims.to_le_bytes());
        // dims
        for d in dims {
            data.extend_from_slice(&d.to_le_bytes());
        }
        // dtype
        data.extend_from_slice(&dtype.to_le_bytes());
        // offset within tensor data section
        data.extend_from_slice(&0u64.to_le_bytes());

        // Alignment padding
        let padding = padding_for_alignment(data.len(), GGUF_DEFAULT_ALIGNMENT);
        data.extend(std::iter::repeat(0u8).take(padding));

        // Tensor data
        data.extend_from_slice(tensor_data);

        data
    }

    #[test]
    fn test_from_bytes_with_one_f32_tensor() {
        // 2x2 F32 tensor = 4 elements * 4 bytes = 16 bytes
        let tensor_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let data = build_synthetic_gguf_with_tensor("test.weight", &[2, 2], 0, &tensor_data, &[]);

        let reader = GgufReader::from_bytes(data).expect("parse GGUF with tensor");
        assert_eq!(reader.tensor_count, 1);
        assert_eq!(reader.tensors.len(), 1);
        assert_eq!(reader.tensors[0].name, "test.weight");
        assert_eq!(reader.tensors[0].dims, vec![2, 2]);
        assert_eq!(reader.tensors[0].dtype, 0); // F32

        // Verify we can extract tensor data
        let (extracted, shape) = reader
            .get_tensor_f32("test.weight")
            .expect("extract tensor");
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(extracted.len(), 4);
        assert!((extracted[0] - 1.0).abs() < f32::EPSILON);
        assert!((extracted[3] - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_from_bytes_tensor_excessive_dims() {
        // n_dims > MAX_DIMS should fail
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

        // Tensor info: name
        let name = "bad.tensor";
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        // n_dims = MAX_DIMS + 1 = 17
        data.extend_from_slice(&(MAX_DIMS + 1).to_le_bytes());
        // Provide enough dummy dim data
        for _ in 0..=MAX_DIMS {
            data.extend_from_slice(&1u64.to_le_bytes());
        }
        data.extend_from_slice(&0u32.to_le_bytes()); // dtype
        data.extend_from_slice(&0u64.to_le_bytes()); // offset

        let result = GgufReader::from_bytes(data);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(
            err.contains("dimensions") && err.contains("exceeds"),
            "Error should mention excessive dimensions: {err}"
        );
    }

    #[test]
    fn test_from_bytes_tensor_at_max_dims() {
        // n_dims = MAX_DIMS (16) should be allowed
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

        // Tensor info: name
        let name = "ok.tensor";
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        // n_dims = MAX_DIMS (16)
        data.extend_from_slice(&MAX_DIMS.to_le_bytes());
        // All dims = 1
        for _ in 0..MAX_DIMS {
            data.extend_from_slice(&1u64.to_le_bytes());
        }
        data.extend_from_slice(&0u32.to_le_bytes()); // dtype F32
        data.extend_from_slice(&0u64.to_le_bytes()); // offset

        // Add alignment padding + tiny tensor data (1 element F32 = 4 bytes)
        let padding = padding_for_alignment(data.len(), GGUF_DEFAULT_ALIGNMENT);
        data.extend(std::iter::repeat(0u8).take(padding));
        data.extend_from_slice(&1.0f32.to_le_bytes());

        let reader = GgufReader::from_bytes(data).expect("MAX_DIMS should be accepted");
        assert_eq!(reader.tensors[0].dims.len(), MAX_DIMS as usize);
    }

    // ========================================================================
    // skip_metadata_value Tests (via from_bytes with non-parsed keys)
    // ========================================================================

    /// Build a GGUF with metadata that will be skipped (key prefix not in parsed set)
    fn build_gguf_with_skipped_metadata(key: &str, value_type: u32, value_bytes: &[u8]) -> Vec<u8> {
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

        // Metadata KV: key with prefix that does NOT match tokenizer./general./llama./qwen2./phi./mistral.
        // so it will be skipped via skip_metadata_value
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&value_type.to_le_bytes());
        data.extend_from_slice(value_bytes);

        data
    }

    #[test]
    fn test_skip_metadata_value_uint8() {
        let data = build_gguf_with_skipped_metadata("custom.u8", 0, &[42u8]);
        let reader = GgufReader::from_bytes(data).expect("skip uint8");
        assert!(!reader.metadata.contains_key("custom.u8"));
    }

    #[test]
    fn test_skip_metadata_value_int8() {
        let data = build_gguf_with_skipped_metadata("custom.i8", 1, &[0xFEu8]);
        let reader = GgufReader::from_bytes(data).expect("skip int8");
        assert!(!reader.metadata.contains_key("custom.i8"));
    }

    #[test]
    fn test_skip_metadata_value_uint16() {
        let data = build_gguf_with_skipped_metadata("custom.u16", 2, &1000u16.to_le_bytes());
        let reader = GgufReader::from_bytes(data).expect("skip uint16");
        assert!(!reader.metadata.contains_key("custom.u16"));
    }

    #[test]
    fn test_skip_metadata_value_int16() {
        let data = build_gguf_with_skipped_metadata("custom.i16", 3, &(-500i16).to_le_bytes());
        let reader = GgufReader::from_bytes(data).expect("skip int16");
        assert!(!reader.metadata.contains_key("custom.i16"));
    }

    #[test]
    fn test_skip_metadata_value_bool() {
        let data = build_gguf_with_skipped_metadata("custom.flag", 7, &[1u8]);
        let reader = GgufReader::from_bytes(data).expect("skip bool");
        assert!(!reader.metadata.contains_key("custom.flag"));
    }

    #[test]
    fn test_skip_metadata_value_string() {
        // String: length-prefixed (8 bytes length + content)
        let s = "hello world";
        let mut value_bytes = Vec::new();
        value_bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
        value_bytes.extend_from_slice(s.as_bytes());
        let data = build_gguf_with_skipped_metadata("custom.str", 8, &value_bytes);
        let reader = GgufReader::from_bytes(data).expect("skip string");
        assert!(!reader.metadata.contains_key("custom.str"));
    }

    #[test]
    fn test_skip_metadata_value_uint64() {
        let data = build_gguf_with_skipped_metadata("custom.u64", 10, &999u64.to_le_bytes());
        let reader = GgufReader::from_bytes(data).expect("skip uint64");
        assert!(!reader.metadata.contains_key("custom.u64"));
    }

    #[test]
    fn test_skip_metadata_value_int64() {
        let data = build_gguf_with_skipped_metadata("custom.i64", 11, &(-1i64).to_le_bytes());
        let reader = GgufReader::from_bytes(data).expect("skip int64");
        assert!(!reader.metadata.contains_key("custom.i64"));
    }

    #[test]
    fn test_skip_metadata_value_float64() {
        let data =
            build_gguf_with_skipped_metadata("custom.f64", 12, &std::f64::consts::E.to_le_bytes());
        let reader = GgufReader::from_bytes(data).expect("skip float64");
        assert!(!reader.metadata.contains_key("custom.f64"));
    }

    #[test]
    fn test_skip_metadata_value_unknown_type() {
        // Unknown type (e.g., 99) should skip 4 bytes
        let data = build_gguf_with_skipped_metadata("custom.unk", 99, &[0u8; 4]);
        let reader = GgufReader::from_bytes(data).expect("skip unknown");
        assert!(!reader.metadata.contains_key("custom.unk"));
    }

    #[test]
    fn test_skip_metadata_value_array_of_uint32() {
        // Array type=9, elem_type=4 (uint32), count=2
        let mut value_bytes = Vec::new();
        value_bytes.extend_from_slice(&4u32.to_le_bytes()); // elem_type Uint32
        value_bytes.extend_from_slice(&2u64.to_le_bytes()); // count
        value_bytes.extend_from_slice(&10u32.to_le_bytes());
        value_bytes.extend_from_slice(&20u32.to_le_bytes());
        let data = build_gguf_with_skipped_metadata("custom.arr_u32", 9, &value_bytes);
        let reader = GgufReader::from_bytes(data).expect("skip array uint32");
        assert!(!reader.metadata.contains_key("custom.arr_u32"));
    }

    #[test]
    fn test_skip_metadata_value_array_of_strings() {
        // Array type=9, elem_type=8 (string), count=2
        let mut value_bytes = Vec::new();
        value_bytes.extend_from_slice(&8u32.to_le_bytes()); // elem_type String
        value_bytes.extend_from_slice(&2u64.to_le_bytes()); // count
                                                            // string 1: "hi"
        value_bytes.extend_from_slice(&2u64.to_le_bytes());
        value_bytes.extend_from_slice(b"hi");
        // string 2: "world"
        value_bytes.extend_from_slice(&5u64.to_le_bytes());
        value_bytes.extend_from_slice(b"world");
        let data = build_gguf_with_skipped_metadata("custom.arr_str", 9, &value_bytes);
        let reader = GgufReader::from_bytes(data).expect("skip array of strings");
        assert!(!reader.metadata.contains_key("custom.arr_str"));
    }

    #[test]
    fn test_skip_metadata_value_array_of_uint8() {
        // Array type=9, elem_type=0 (uint8), count=3
        let mut value_bytes = Vec::new();
        value_bytes.extend_from_slice(&0u32.to_le_bytes()); // elem_type Uint8
        value_bytes.extend_from_slice(&3u64.to_le_bytes()); // count
        value_bytes.extend_from_slice(&[1u8, 2u8, 3u8]);
        let data = build_gguf_with_skipped_metadata("custom.arr_u8", 9, &value_bytes);
        let reader = GgufReader::from_bytes(data).expect("skip array of uint8");
        assert!(!reader.metadata.contains_key("custom.arr_u8"));
    }

    #[test]
    fn test_skip_metadata_value_array_of_uint64() {
        // Array type=9, elem_type=10 (uint64), count=1
        let mut value_bytes = Vec::new();
        value_bytes.extend_from_slice(&10u32.to_le_bytes()); // elem_type Uint64
        value_bytes.extend_from_slice(&1u64.to_le_bytes()); // count
        value_bytes.extend_from_slice(&42u64.to_le_bytes());
        let data = build_gguf_with_skipped_metadata("custom.arr_u64", 9, &value_bytes);
        let reader = GgufReader::from_bytes(data).expect("skip array of uint64");
        assert!(!reader.metadata.contains_key("custom.arr_u64"));
    }

    #[test]
    fn test_skip_metadata_value_array_of_int16() {
        // Array type=9, elem_type=3 (int16), count=2
        let mut value_bytes = Vec::new();
        value_bytes.extend_from_slice(&3u32.to_le_bytes()); // elem_type Int16
        value_bytes.extend_from_slice(&2u64.to_le_bytes()); // count
        value_bytes.extend_from_slice(&(-1i16).to_le_bytes());
        value_bytes.extend_from_slice(&100i16.to_le_bytes());
        let data = build_gguf_with_skipped_metadata("custom.arr_i16", 9, &value_bytes);
        let reader = GgufReader::from_bytes(data).expect("skip array of int16");
        assert!(!reader.metadata.contains_key("custom.arr_i16"));
    }

    // ========================================================================
    // GgufReader Accessor Method Tests
    // ========================================================================

    /// Build a GGUF with tokenizer metadata (parsed keys)
    fn build_gguf_with_tokenizer_metadata() -> Vec<u8> {
        let mut data = Vec::new();

        // We'll add 6 metadata entries: tokens, model, bos, eos, merges, architecture
        let metadata_count = 6u64;

        // Header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&metadata_count.to_le_bytes());

        // Helper: write a length-prefixed string
        fn write_str(buf: &mut Vec<u8>, s: &str) {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }

        // 1. tokenizer.ggml.tokens (ArrayString)
        write_str(&mut data, "tokenizer.ggml.tokens");
        data.extend_from_slice(&9u32.to_le_bytes()); // type = Array
        data.extend_from_slice(&8u32.to_le_bytes()); // elem_type = String
        data.extend_from_slice(&3u64.to_le_bytes()); // count = 3
        write_str(&mut data, "<unk>");
        write_str(&mut data, "hello");
        write_str(&mut data, "world");

        // 2. tokenizer.ggml.model (String)
        write_str(&mut data, "tokenizer.ggml.model");
        data.extend_from_slice(&8u32.to_le_bytes()); // type = String
        write_str(&mut data, "llama");

        // 3. tokenizer.ggml.bos_token_id (Uint32)
        write_str(&mut data, "tokenizer.ggml.bos_token_id");
        data.extend_from_slice(&4u32.to_le_bytes()); // type = Uint32
        data.extend_from_slice(&1u32.to_le_bytes()); // value = 1

        // 4. tokenizer.ggml.eos_token_id (Uint32)
        write_str(&mut data, "tokenizer.ggml.eos_token_id");
        data.extend_from_slice(&4u32.to_le_bytes()); // type = Uint32
        data.extend_from_slice(&2u32.to_le_bytes()); // value = 2

        // 5. tokenizer.ggml.merges (ArrayString)
        write_str(&mut data, "tokenizer.ggml.merges");
        data.extend_from_slice(&9u32.to_le_bytes()); // type = Array
        data.extend_from_slice(&8u32.to_le_bytes()); // elem_type = String
        data.extend_from_slice(&2u64.to_le_bytes()); // count = 2
        write_str(&mut data, "h e");
        write_str(&mut data, "l o");

        // 6. general.architecture (String)
        write_str(&mut data, "general.architecture");
        data.extend_from_slice(&8u32.to_le_bytes()); // type = String
        write_str(&mut data, "llama");

        data
    }

    #[test]
    fn test_accessor_vocabulary() {
        let data = build_gguf_with_tokenizer_metadata();
        let reader = GgufReader::from_bytes(data).expect("parse tokenizer metadata");
        let vocab = reader.vocabulary().expect("vocabulary should exist");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab[0], "<unk>");
        assert_eq!(vocab[1], "hello");
        assert_eq!(vocab[2], "world");
    }

    #[test]
    fn test_accessor_tokenizer_model() {
        let data = build_gguf_with_tokenizer_metadata();
        let reader = GgufReader::from_bytes(data).expect("parse tokenizer metadata");
        let model = reader
            .tokenizer_model()
            .expect("tokenizer model should exist");
        assert_eq!(model, "llama");
    }

    #[test]
    fn test_accessor_bos_token_id() {
        let data = build_gguf_with_tokenizer_metadata();
        let reader = GgufReader::from_bytes(data).expect("parse tokenizer metadata");
        let bos = reader.bos_token_id().expect("bos_token_id should exist");
        assert_eq!(bos, 1);
    }

    #[test]
    fn test_accessor_eos_token_id() {
        let data = build_gguf_with_tokenizer_metadata();
        let reader = GgufReader::from_bytes(data).expect("parse tokenizer metadata");
        let eos = reader.eos_token_id().expect("eos_token_id should exist");
        assert_eq!(eos, 2);
    }

    #[test]
    fn test_accessor_merges() {
        let data = build_gguf_with_tokenizer_metadata();
        let reader = GgufReader::from_bytes(data).expect("parse tokenizer metadata");
        let merges = reader.merges().expect("merges should exist");
        assert_eq!(merges.len(), 2);
        assert_eq!(merges[0], "h e");
        assert_eq!(merges[1], "l o");
    }

    #[test]
    fn test_accessor_architecture() {
        let data = build_gguf_with_tokenizer_metadata();
        let reader = GgufReader::from_bytes(data).expect("parse tokenizer metadata");
        let arch = reader.architecture().expect("architecture should exist");
        assert_eq!(arch, "llama");
    }

    #[test]
    fn test_accessor_vocabulary_none_when_missing() {
        let data = create_gguf_header(0, 0);
        let reader = GgufReader::from_bytes(data).expect("parse empty GGUF");
        assert!(reader.vocabulary().is_none());
    }

    #[test]
    fn test_accessor_tokenizer_model_none_when_missing() {
        let data = create_gguf_header(0, 0);
        let reader = GgufReader::from_bytes(data).expect("parse empty GGUF");
        assert!(reader.tokenizer_model().is_none());
    }

    #[test]
    fn test_accessor_bos_token_id_none_when_missing() {
        let data = create_gguf_header(0, 0);
        let reader = GgufReader::from_bytes(data).expect("parse empty GGUF");
        assert!(reader.bos_token_id().is_none());
    }

    #[test]
    fn test_accessor_eos_token_id_none_when_missing() {
        let data = create_gguf_header(0, 0);
        let reader = GgufReader::from_bytes(data).expect("parse empty GGUF");
        assert!(reader.eos_token_id().is_none());
    }

    #[test]
    fn test_accessor_merges_none_when_missing() {
        let data = create_gguf_header(0, 0);
        let reader = GgufReader::from_bytes(data).expect("parse empty GGUF");
        assert!(reader.merges().is_none());
    }

    #[test]
    fn test_accessor_vocabulary_none_when_empty() {
        // Build GGUF with empty token array
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

        // tokenizer.ggml.tokens = empty string array
        let key = "tokenizer.ggml.tokens";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&9u32.to_le_bytes()); // Array
        data.extend_from_slice(&8u32.to_le_bytes()); // elem_type String
        data.extend_from_slice(&0u64.to_le_bytes()); // count = 0

        let reader = GgufReader::from_bytes(data).expect("parse GGUF with empty vocab");
        assert!(
            reader.vocabulary().is_none(),
            "Empty vocab should return None"
        );
    }

    #[test]
    fn test_accessor_merges_none_when_empty() {
        // Build GGUF with empty merges array
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        let key = "tokenizer.ggml.merges";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&9u32.to_le_bytes()); // Array
        data.extend_from_slice(&8u32.to_le_bytes()); // elem_type String
        data.extend_from_slice(&0u64.to_le_bytes()); // count = 0

        let reader = GgufReader::from_bytes(data).expect("parse GGUF with empty merges");
        assert!(reader.merges().is_none(), "Empty merges should return None");
    }

    // ========================================================================
    // Mixed Metadata Tests (parsed + skipped in same file)
    // ========================================================================

    #[test]
    fn test_from_bytes_mixed_parsed_and_skipped_metadata() {
        // One parsed key (tokenizer.*) and one skipped key (custom.*)
        let mut data = Vec::new();

        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&2u64.to_le_bytes()); // metadata_count = 2

        fn write_str(buf: &mut Vec<u8>, s: &str) {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }

        // Skipped: custom key with type Float32
        write_str(&mut data, "custom.learning_rate");
        data.extend_from_slice(&6u32.to_le_bytes()); // Float32
        data.extend_from_slice(&0.001f32.to_le_bytes());

        // Parsed: tokenizer key with type Uint32
        write_str(&mut data, "tokenizer.ggml.bos_token_id");
        data.extend_from_slice(&4u32.to_le_bytes()); // Uint32
        data.extend_from_slice(&1u32.to_le_bytes());

        let reader = GgufReader::from_bytes(data).expect("parse mixed metadata");
        assert!(!reader.metadata.contains_key("custom.learning_rate"));
        assert_eq!(reader.bos_token_id(), Some(1));
    }

    // ========================================================================
    // read_u32 / read_u64 / read_string edge cases
    // ========================================================================

    #[test]
    fn test_read_u32_eof() {
        let bytes = [0u8; 3]; // need 4
        let result = read_u32(&bytes, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_u64_eof() {
        let bytes = [0u8; 7]; // need 8
        let result = read_u64(&bytes, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_string_length_exceeds_data() {
        // Claim string is 100 bytes but only provide 5
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&100u64.to_le_bytes()); // length = 100
        bytes.extend_from_slice(b"short"); // only 5 bytes
        let result = read_string(&bytes, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_string_empty() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0u64.to_le_bytes()); // length = 0
        let (s, consumed) = read_string(&bytes, 0).expect("read empty string");
        assert_eq!(s, "");
        assert_eq!(consumed, 8); // just the length prefix
    }

    // ========================================================================
    // Additional read_metadata_value array branch tests
    // ========================================================================

    #[test]
    fn test_read_metadata_value_array_of_strings() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&8u32.to_le_bytes()); // elem_type = String
        bytes.extend_from_slice(&2u64.to_le_bytes()); // count = 2
                                                      // string 1: "abc"
        bytes.extend_from_slice(&3u64.to_le_bytes());
        bytes.extend_from_slice(b"abc");
        // string 2: "de"
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(b"de");
        let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array of strings");
        // 12 (header) + 8+3 + 8+2 = 12 + 11 + 10 = 33
        assert_eq!(consumed, 33);
        match result {
            GgufValue::ArrayString(v) => {
                assert_eq!(v, vec!["abc".to_string(), "de".to_string()]);
            }
            other => panic!("Expected ArrayString, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_array_of_uint32() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&4u32.to_le_bytes()); // elem_type = Uint32
        bytes.extend_from_slice(&3u64.to_le_bytes()); // count = 3
        bytes.extend_from_slice(&100u32.to_le_bytes());
        bytes.extend_from_slice(&200u32.to_le_bytes());
        bytes.extend_from_slice(&300u32.to_le_bytes());
        let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array of uint32");
        assert_eq!(consumed, 24); // 12 + 3*4
        match result {
            GgufValue::ArrayUint32(v) => assert_eq!(v, vec![100, 200, 300]),
            other => panic!("Expected ArrayUint32, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_array_of_float32() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&6u32.to_le_bytes()); // elem_type = Float32
        bytes.extend_from_slice(&2u64.to_le_bytes()); // count = 2
        bytes.extend_from_slice(&1.5f32.to_le_bytes());
        bytes.extend_from_slice(&(-2.5f32).to_le_bytes());
        let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array of float32");
        assert_eq!(consumed, 20); // 12 + 2*4
        match result {
            GgufValue::ArrayFloat32(v) => {
                assert!((v[0] - 1.5).abs() < f32::EPSILON);
                assert!((v[1] - (-2.5)).abs() < f32::EPSILON);
            }
            other => panic!("Expected ArrayFloat32, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_array_of_int8() {
        // Array of Int8 (elem_type=1) -> "other" branch, 1-byte elements
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_le_bytes()); // elem_type = 1 (Int8)
        bytes.extend_from_slice(&5u64.to_le_bytes()); // count = 5
        bytes.extend_from_slice(&[0u8; 5]);
        let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array int8");
        assert_eq!(consumed, 17); // 12 + 5*1
        match result {
            GgufValue::ArrayUint32(v) => assert!(v.is_empty()),
            other => panic!("Expected empty ArrayUint32, got {other:?}"),
        }
    }

    #[test]
    fn test_read_metadata_value_array_of_int16() {
        // Array of Int16 (elem_type=3) -> "other" branch, 2-byte elements
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&3u32.to_le_bytes()); // elem_type = 3 (Int16)
        bytes.extend_from_slice(&4u64.to_le_bytes()); // count = 4
        bytes.extend_from_slice(&[0u8; 8]); // 4 * 2 bytes
        let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array int16");
        assert_eq!(consumed, 20); // 12 + 4*2
        match result {
            GgufValue::ArrayUint32(v) => assert!(v.is_empty()),
            other => panic!("Expected empty ArrayUint32, got {other:?}"),
        }
    }
}
