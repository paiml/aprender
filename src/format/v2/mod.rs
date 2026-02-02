//! APR Format Module (GH-119)
//!
//! Implements the APR format specification with:
//! - 64-byte tensor alignment for zero-copy mmap
//! - LZ4 block compression (64KB blocks)
//! - JSON metadata section
//! - Multi-file sharding for 10B+ parameter models
//! - Single unified format (no versioning complexity)
//!
//! # Format Structure (APR)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Header (64 bytes, 64-byte aligned)                          │
//! │   - Magic: "APR\0" (4 bytes) - ONE format, no versioning    │
//! │   - Version: major.minor (2 bytes)                          │
//! │   - Flags (2 bytes)                                         │
//! │   - Tensor count (4 bytes)                                  │
//! │   - Metadata offset (8 bytes)                               │
//! │   - Metadata size (4 bytes)                                 │
//! │   - Tensor index offset (8 bytes)                           │
//! │   - Data offset (8 bytes)                                   │
//! │   - Checksum (4 bytes)                                      │
//! │   - Reserved (20 bytes, zero-padded)                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │ JSON Metadata (variable, padded to 64-byte boundary)        │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Tensor Index (sorted by name, 64-byte aligned entries)      │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Tensor Data (each tensor 64-byte aligned)                   │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Footer Checksum (4 bytes)                                   │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust
//! use aprender::format::v2::{AprV2Header, AprV2Flags, MAGIC_V2, ALIGNMENT};
//!
//! let header = AprV2Header::new();
//! assert_eq!(header.magic, MAGIC_V2);
//! assert!(header.is_valid());
//! ```
//!
//! # PMAT Compliance
//!
//! - Zero `unwrap()` calls
//! - All public APIs return `Result<T, E>`
//! - 64-byte alignment enforced at type level

use crate::format::f16_safety::F16_MIN_NORMAL;
use crate::format::gguf::dequant::{dequantize_q4_k, dequantize_q6_k};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};

// ============================================================================
// CRC32 (IEEE polynomial, matching format/mod.rs)
// ============================================================================

/// CRC32 checksum (IEEE polynomial 0xEDB88320)
fn crc32(data: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut crc = i as u32;
            let mut j = 0;
            while j < 8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB8_8320;
                } else {
                    crc >>= 1;
                }
                j += 1;
            }
            table[i] = crc;
            i += 1;
        }
        table
    };

    let mut crc = 0xFFFF_FFFF_u32;
    for &byte in data {
        let idx = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ TABLE[idx];
    }
    !crc
}

// ============================================================================
// IEEE 754 Half-Precision (f16) Conversion
// ============================================================================

/// Convert f32 to f16 (IEEE 754 half-precision)
///
/// Half-precision format:
/// - Sign: 1 bit
/// - Exponent: 5 bits (bias 15)
/// - Mantissa: 10 bits
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (bits >> 23) & 0xFF;
    let mantissa = bits & 0x7F_FFFF;

    if exp == 0 {
        // Zero or denormal f32 → zero f16
        sign
    } else if exp == 255 {
        // Inf or NaN
        if mantissa == 0 {
            sign | 0x7C00 // Inf
        } else {
            sign | 0x7E00 // NaN (quiet)
        }
    } else {
        // Normalized number
        let new_exp = (exp as i32) - 127 + 15;

        if new_exp >= 31 {
            // Overflow to infinity
            sign | 0x7C00
        } else if new_exp <= 0 {
            // Underflow to zero (could do denormals but not worth it)
            sign
        } else {
            let new_mantissa = (mantissa >> 13) as u16;
            sign | ((new_exp as u16) << 10) | new_mantissa
        }
    }
}

/// Convert f16 to f32
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = (bits >> 10) & 0x1F;
    let mantissa = u32::from(bits & 0x3FF);

    if exp == 0 {
        if mantissa == 0 {
            f32::from_bits(sign)
        } else {
            // Denormal f16 → normalized f32
            let mut m = mantissa;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            let new_exp = (127 - 15 + 1 + e) as u32;
            let new_mantissa = (m & 0x3FF) << 13;
            f32::from_bits(sign | (new_exp << 23) | new_mantissa)
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits(sign | 0x7F80_0000 | (mantissa << 13))
    } else {
        // Normalized
        let new_exp = (u32::from(exp) - 15 + 127) << 23;
        let new_mantissa = mantissa << 13;
        f32::from_bits(sign | new_exp | new_mantissa)
    }
}

/// Dequantize Q4 block-quantized data to f32
///
/// Format: blocks of [scale: f16 (2 bytes)] + [packed nibbles: 16 bytes]
/// Each block contains 32 values.
fn dequantize_q4(data: &[u8], element_count: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;

    let mut result = Vec::with_capacity(element_count);
    let mut pos = 0;
    let mut remaining = element_count;

    while remaining > 0 && pos + 2 <= data.len() {
        // Read scale (f16)
        // GH-186 FIX: Clamp NaN/Inf/subnormal to prevent propagation
        // Uses shared F16_MIN_NORMAL from crate::format::f16_safety (P2 fix)
        let scale_bits = u16::from_le_bytes([data[pos], data[pos + 1]]);
        let scale_raw = f16_to_f32(scale_bits);
        let scale = if scale_raw.is_nan() || scale_raw.is_infinite() || scale_raw.abs() < F16_MIN_NORMAL {
            0.0
        } else {
            scale_raw
        };
        pos += 2;

        // Read packed nibbles (16 bytes max)
        let values_in_block = remaining.min(BLOCK_SIZE);

        for i in 0..values_in_block {
            let byte_idx = pos + i / 2;
            if byte_idx >= data.len() {
                break;
            }

            let byte = data[byte_idx];
            let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };

            // Convert back from unsigned nibble (0-15) to signed (-8 to 7)
            let q = (nibble as i8) - 8;
            result.push(f32::from(q) * scale);
        }

        // Move to next block (always 18 bytes per block in storage)
        pos += 16;
        remaining = remaining.saturating_sub(BLOCK_SIZE);
    }

    // Pad with zeros if needed (partial last block)
    result.resize(element_count, 0.0);
    result
}

// ============================================================================
// Constants
// ============================================================================

/// APR magic number: "APR\0" in ASCII (0x41505200)
/// ONE format. No versioning. Period.
pub const MAGIC_V2: [u8; 4] = [0x41, 0x50, 0x52, 0x00];

/// Format version 2.0
pub const VERSION_V2: (u8, u8) = (2, 0);

/// Header size in bytes (64-byte aligned)
pub const HEADER_SIZE_V2: usize = 64;

/// Tensor alignment in bytes (for zero-copy mmap)
pub const ALIGNMENT: usize = 64;

/// LZ4 block size in bytes
pub const LZ4_BLOCK_SIZE: usize = 64 * 1024; // 64KB

/// Maximum metadata size (16MB)
pub const MAX_METADATA_SIZE: usize = 16 * 1024 * 1024;

/// Maximum tensor name length
pub const MAX_TENSOR_NAME_LEN: usize = 256;

// ============================================================================
// Flags
// ============================================================================

/// APR v2 feature flags (16-bit for expanded feature set)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AprV2Flags(u16);

impl AprV2Flags {
    /// Payload is compressed with LZ4
    pub const LZ4_COMPRESSED: u16 = 0b0000_0000_0000_0001;
    /// Payload is compressed with Zstd
    pub const ZSTD_COMPRESSED: u16 = 0b0000_0000_0000_0010;
    /// Payload is encrypted (AES-256-GCM)
    pub const ENCRYPTED: u16 = 0b0000_0000_0000_0100;
    /// Has digital signature (Ed25519)
    pub const SIGNED: u16 = 0b0000_0000_0000_1000;
    /// Model is sharded across multiple files
    pub const SHARDED: u16 = 0b0000_0000_0001_0000;
    /// Tensors are quantized
    pub const QUANTIZED: u16 = 0b0000_0000_0010_0000;
    /// Has embedded filterbank (for Whisper models)
    pub const HAS_FILTERBANK: u16 = 0b0000_0000_0100_0000;
    /// Has model card metadata
    pub const HAS_MODEL_CARD: u16 = 0b0000_0000_1000_0000;
    /// Supports streaming/chunked loading
    pub const STREAMING: u16 = 0b0000_0001_0000_0000;
    /// Contains vocabulary/tokenizer
    pub const HAS_VOCAB: u16 = 0b0000_0010_0000_0000;

    /// Create new empty flags
    #[must_use]
    pub const fn new() -> Self {
        Self(0)
    }

    /// Create from raw u16 value
    #[must_use]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Get raw bits
    #[must_use]
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// Check if flag is set
    #[must_use]
    pub const fn contains(self, flag: u16) -> bool {
        (self.0 & flag) == flag
    }

    /// Set a flag
    #[must_use]
    pub const fn with(self, flag: u16) -> Self {
        Self(self.0 | flag)
    }

    /// Clear a flag
    #[must_use]
    pub const fn without(self, flag: u16) -> Self {
        Self(self.0 & !flag)
    }

    /// Check if LZ4 compressed
    #[must_use]
    pub const fn is_lz4_compressed(self) -> bool {
        self.contains(Self::LZ4_COMPRESSED)
    }

    /// Check if Zstd compressed
    #[must_use]
    pub const fn is_zstd_compressed(self) -> bool {
        self.contains(Self::ZSTD_COMPRESSED)
    }

    /// Check if encrypted
    #[must_use]
    pub const fn is_encrypted(self) -> bool {
        self.contains(Self::ENCRYPTED)
    }

    /// Check if sharded
    #[must_use]
    pub const fn is_sharded(self) -> bool {
        self.contains(Self::SHARDED)
    }

    /// Check if quantized
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        self.contains(Self::QUANTIZED)
    }
}

// ============================================================================
// Header
// ============================================================================

/// APR file header (64 bytes)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AprV2Header {
    /// Magic number ("APR\0") - ONE format, no versioning
    pub magic: [u8; 4],
    /// Format version (major, minor)
    pub version: (u8, u8),
    /// Feature flags
    pub flags: AprV2Flags,
    /// Number of tensors
    pub tensor_count: u32,
    /// Offset to JSON metadata section
    pub metadata_offset: u64,
    /// Size of metadata in bytes
    pub metadata_size: u32,
    /// Offset to tensor index
    pub tensor_index_offset: u64,
    /// Offset to tensor data
    pub data_offset: u64,
    /// Header checksum (CRC32)
    pub checksum: u32,
    /// Reserved for future use (zero-padded)
    pub reserved: [u8; 20],
}

impl Default for AprV2Header {
    fn default() -> Self {
        Self::new()
    }
}

impl AprV2Header {
    /// Create new v2 header with defaults
    #[must_use]
    pub fn new() -> Self {
        Self {
            magic: MAGIC_V2,
            version: VERSION_V2,
            flags: AprV2Flags::new(),
            tensor_count: 0,
            metadata_offset: HEADER_SIZE_V2 as u64,
            metadata_size: 0,
            tensor_index_offset: 0,
            data_offset: 0,
            checksum: 0,
            reserved: [0u8; 20],
        }
    }

    /// Check if header has valid magic number
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.magic == MAGIC_V2
    }

    /// Serialize header to bytes
    #[must_use]
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE_V2] {
        let mut buf = [0u8; HEADER_SIZE_V2];

        buf[0..4].copy_from_slice(&self.magic);
        buf[4] = self.version.0;
        buf[5] = self.version.1;
        buf[6..8].copy_from_slice(&self.flags.bits().to_le_bytes());
        buf[8..12].copy_from_slice(&self.tensor_count.to_le_bytes());
        buf[12..20].copy_from_slice(&self.metadata_offset.to_le_bytes());
        buf[20..24].copy_from_slice(&self.metadata_size.to_le_bytes());
        buf[24..32].copy_from_slice(&self.tensor_index_offset.to_le_bytes());
        buf[32..40].copy_from_slice(&self.data_offset.to_le_bytes());
        buf[40..44].copy_from_slice(&self.checksum.to_le_bytes());
        buf[44..64].copy_from_slice(&self.reserved);

        buf
    }

    /// Deserialize header from bytes
    ///
    /// # Errors
    /// Returns error if buffer is too small or magic is invalid.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, V2FormatError> {
        if buf.len() < HEADER_SIZE_V2 {
            return Err(V2FormatError::InvalidHeader("buffer too small".to_string()));
        }

        let magic: [u8; 4] = buf[0..4]
            .try_into()
            .map_err(|_| V2FormatError::InvalidHeader("failed to read magic".to_string()))?;

        // Check for v2 magic only
        if magic != MAGIC_V2 {
            return Err(V2FormatError::InvalidMagic(magic));
        }

        let version = (buf[4], buf[5]);
        let flags = AprV2Flags::from_bits(u16::from_le_bytes([buf[6], buf[7]]));
        let tensor_count = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let metadata_offset = u64::from_le_bytes(buf[12..20].try_into().unwrap_or([0; 8]));
        let metadata_size = u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]);
        let tensor_index_offset = u64::from_le_bytes(buf[24..32].try_into().unwrap_or([0; 8]));
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap_or([0; 8]));
        let checksum = u32::from_le_bytes([buf[40], buf[41], buf[42], buf[43]]);

        let mut reserved = [0u8; 20];
        reserved.copy_from_slice(&buf[44..64]);

        Ok(Self {
            magic,
            version,
            flags,
            tensor_count,
            metadata_offset,
            metadata_size,
            tensor_index_offset,
            data_offset,
            checksum,
            reserved,
        })
    }

    /// Compute header checksum (CRC32 of header bytes excluding checksum field)
    #[must_use]
    pub fn compute_checksum(&self) -> u32 {
        let bytes = self.to_bytes();
        // Exclude checksum field (bytes 40-43) from calculation
        // Concatenate the two regions and compute CRC32
        let mut data = Vec::with_capacity(60);
        data.extend_from_slice(&bytes[0..40]);
        data.extend_from_slice(&bytes[44..64]);
        crc32(&data)
    }

    /// Update checksum field
    pub fn update_checksum(&mut self) {
        self.checksum = self.compute_checksum();
    }

    /// Verify header checksum
    #[must_use]
    pub fn verify_checksum(&self) -> bool {
        self.checksum == self.compute_checksum()
    }
}

// ============================================================================
// Metadata
// ============================================================================

/// APR v2 JSON metadata section
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AprV2Metadata {
    /// Model type identifier
    #[serde(default)]
    pub model_type: String,

    /// Model name
    #[serde(default)]
    pub name: Option<String>,

    /// Model description
    #[serde(default)]
    pub description: Option<String>,

    /// Model author/organization
    #[serde(default)]
    pub author: Option<String>,

    /// Model license
    #[serde(default)]
    pub license: Option<String>,

    /// Model version string
    #[serde(default)]
    pub version: Option<String>,

    /// Source/provenance URI (DD6: Model provenance tracking)
    /// Examples: "<hf://openai/whisper-tiny>", "<local://path/to/model.safetensors>"
    #[serde(default)]
    pub source: Option<String>,

    /// Original format before conversion
    /// Examples: "safetensors", "gguf", "pytorch"
    #[serde(default)]
    pub original_format: Option<String>,

    /// Creation timestamp (ISO 8601)
    #[serde(default)]
    pub created_at: Option<String>,

    /// Total model size in bytes
    #[serde(default)]
    pub total_size: u64,

    /// Parameter count
    #[serde(default)]
    pub param_count: u64,

    /// Quantization info
    #[serde(default)]
    pub quantization: Option<QuantizationMetadata>,

    /// Shard info (for multi-file models)
    #[serde(default)]
    pub sharding: Option<ShardingMetadata>,

    /// Chat template (Jinja2 format, from tokenizer_config.json)
    /// Per spec: chat-template-improvement-spec.md CTA-01
    #[serde(default)]
    pub chat_template: Option<String>,

    /// Detected chat template format
    /// Per spec: chat-template-improvement-spec.md CTA-03
    /// Values: "chatml", "llama2", "mistral", "phi", "alpaca", "custom", "raw"
    #[serde(default)]
    pub chat_format: Option<String>,

    /// Special tokens for chat templates
    /// Per spec: chat-template-improvement-spec.md CTA-04
    #[serde(default)]
    pub special_tokens: Option<ChatSpecialTokens>,

    // ========================================================================
    // Transformer Config (CRITICAL for inference - realizar::apr::AprMetadata)
    // ========================================================================
    /// Model architecture family (e.g., "llama", "qwen2", "phi")
    #[serde(default)]
    pub architecture: Option<String>,

    /// Hidden dimension size
    #[serde(default)]
    pub hidden_size: Option<usize>,

    /// Number of transformer layers
    #[serde(default)]
    pub num_layers: Option<usize>,

    /// Number of attention heads
    #[serde(default)]
    pub num_heads: Option<usize>,

    /// Number of key-value heads (for GQA, defaults to num_heads)
    #[serde(default)]
    pub num_kv_heads: Option<usize>,

    /// Vocabulary size
    #[serde(default)]
    pub vocab_size: Option<usize>,

    /// FFN intermediate dimension
    #[serde(default)]
    pub intermediate_size: Option<usize>,

    /// Maximum context/sequence length
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    /// RoPE theta for position encoding
    #[serde(default)]
    pub rope_theta: Option<f32>,

    /// RoPE type: 0=NORM (adjacent pairs), 2=NEOX (split halves)
    /// CORRECTNESS-011: Qwen2.5 models require rope_type=2 (NEOX style)
    #[serde(default)]
    pub rope_type: Option<u32>,

    /// Layer norm epsilon
    #[serde(default)]
    pub rms_norm_eps: Option<f32>,

    /// Custom key-value pairs
    #[serde(default, flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

/// Special tokens for chat templates (CTA-04)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatSpecialTokens {
    /// Beginning of sequence token
    #[serde(default)]
    pub bos_token: Option<String>,

    /// End of sequence token
    #[serde(default)]
    pub eos_token: Option<String>,

    /// Unknown token
    #[serde(default)]
    pub unk_token: Option<String>,

    /// Padding token
    #[serde(default)]
    pub pad_token: Option<String>,

    /// ChatML start token (<|im_start|>)
    #[serde(default)]
    pub im_start_token: Option<String>,

    /// ChatML end token (<|im_end|>)
    #[serde(default)]
    pub im_end_token: Option<String>,
}

impl AprV2Metadata {
    /// Create new empty metadata
    #[must_use]
    pub fn new(model_type: impl Into<String>) -> Self {
        Self {
            model_type: model_type.into(),
            ..Default::default()
        }
    }

    /// Serialize to JSON bytes
    ///
    /// # Errors
    /// Returns error if serialization fails.
    pub fn to_json(&self) -> Result<Vec<u8>, V2FormatError> {
        serde_json::to_vec(self).map_err(|e| V2FormatError::MetadataError(e.to_string()))
    }

    /// Serialize to pretty JSON string
    ///
    /// # Errors
    /// Returns error if serialization fails.
    pub fn to_json_pretty(&self) -> Result<String, V2FormatError> {
        serde_json::to_string_pretty(self).map_err(|e| V2FormatError::MetadataError(e.to_string()))
    }

    /// Deserialize from JSON bytes
    ///
    /// # Errors
    /// Returns error if deserialization fails.
    pub fn from_json(data: &[u8]) -> Result<Self, V2FormatError> {
        serde_json::from_slice(data).map_err(|e| V2FormatError::MetadataError(e.to_string()))
    }
}

/// Quantization metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuantizationMetadata {
    /// Quantization type (e.g., "int8", "int4", "fp16")
    pub quant_type: String,
    /// Bits per weight
    pub bits: u8,
    /// Block size for block quantization
    pub block_size: Option<usize>,
    /// Whether symmetric quantization
    pub symmetric: bool,
}

/// Sharding metadata for multi-file models
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShardingMetadata {
    /// Total number of shards
    pub shard_count: usize,
    /// This shard's index (0-based)
    pub shard_index: usize,
    /// Total size across all shards
    pub total_size: u64,
    /// Shard file pattern (e.g., "model-{:05d}-of-{:05d}.apr")
    pub pattern: Option<String>,
}

// ============================================================================
// Tensor Index
// ============================================================================

/// Tensor index entry (fixed size for efficient lookup)
#[derive(Debug, Clone)]
pub struct TensorIndexEntry {
    /// Tensor name (up to 256 bytes)
    pub name: String,
    /// Data type
    pub dtype: TensorDType,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Offset in data section (64-byte aligned)
    pub offset: u64,
    /// Size in bytes
    pub size: u64,
}

impl TensorIndexEntry {
    /// Create new tensor index entry
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        dtype: TensorDType,
        shape: Vec<usize>,
        offset: u64,
        size: u64,
    ) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape,
            offset,
            size,
        }
    }

    /// Calculate element count
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Serialize to bytes
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Name length (2 bytes) + name
        let name_bytes = self.name.as_bytes();
        let name_len = name_bytes.len().min(MAX_TENSOR_NAME_LEN) as u16;
        buf.extend_from_slice(&name_len.to_le_bytes());
        buf.extend_from_slice(&name_bytes[..name_len as usize]);

        // Dtype (1 byte)
        buf.push(self.dtype as u8);

        // Shape: ndim (1 byte) + dims (8 bytes each)
        let ndim = self.shape.len().min(8) as u8;
        buf.push(ndim);
        for &dim in self.shape.iter().take(8) {
            buf.extend_from_slice(&(dim as u64).to_le_bytes());
        }

        // Offset (8 bytes)
        buf.extend_from_slice(&self.offset.to_le_bytes());

        // Size (8 bytes)
        buf.extend_from_slice(&self.size.to_le_bytes());

        buf
    }

    /// Deserialize from bytes
    ///
    /// # Errors
    /// Returns error if buffer is invalid.
    pub fn from_bytes(buf: &[u8]) -> Result<(Self, usize), V2FormatError> {
        if buf.len() < 4 {
            return Err(V2FormatError::InvalidTensorIndex(
                "buffer too small".to_string(),
            ));
        }

        let mut pos = 0;

        // Name length + name
        let name_len = u16::from_le_bytes([buf[pos], buf[pos + 1]]) as usize;
        pos += 2;

        if buf.len() < pos + name_len + 18 {
            return Err(V2FormatError::InvalidTensorIndex(
                "buffer too small for name".to_string(),
            ));
        }

        let name = String::from_utf8_lossy(&buf[pos..pos + name_len]).to_string();
        pos += name_len;

        // Dtype
        let dtype = TensorDType::from_u8(buf[pos])
            .ok_or_else(|| V2FormatError::InvalidTensorIndex("invalid dtype".to_string()))?;
        pos += 1;

        // Shape
        let ndim = buf[pos] as usize;
        pos += 1;

        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            if buf.len() < pos + 8 {
                return Err(V2FormatError::InvalidTensorIndex(
                    "buffer too small for shape".to_string(),
                ));
            }
            let dim = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap_or([0; 8])) as usize;
            shape.push(dim);
            pos += 8;
        }

        // Offset
        if buf.len() < pos + 16 {
            return Err(V2FormatError::InvalidTensorIndex(
                "buffer too small for offset/size".to_string(),
            ));
        }
        let offset = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap_or([0; 8]));
        pos += 8;

        // Size
        let size = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap_or([0; 8]));
        pos += 8;

        Ok((
            Self {
                name,
                dtype,
                shape,
                offset,
                size,
            },
            pos,
        ))
    }
}

/// Tensor data type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TensorDType {
    /// 32-bit float
    F32 = 0,
    /// 16-bit float (half precision)
    F16 = 1,
    /// Brain float 16
    BF16 = 2,
    /// 64-bit float
    F64 = 3,
    /// 32-bit signed integer
    I32 = 4,
    /// 64-bit signed integer
    I64 = 5,
    /// 8-bit signed integer (quantized)
    I8 = 6,
    /// 8-bit unsigned integer
    U8 = 7,
    /// 4-bit quantized (packed, 2 values per byte)
    Q4 = 8,
    /// 8-bit quantized with scale
    Q8 = 9,
    /// GGUF Q4_K format (raw super-blocks, ~4.5 bits/weight)
    /// Format: 256-element blocks with super-block scales
    Q4K = 12,
    /// GGUF Q6_K format (raw super-blocks, ~6.5 bits/weight)
    Q6K = 14,
}

impl TensorDType {
    /// Convert from u8
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::BF16),
            3 => Some(Self::F64),
            4 => Some(Self::I32),
            5 => Some(Self::I64),
            6 => Some(Self::I8),
            7 => Some(Self::U8),
            8 => Some(Self::Q4),
            9 => Some(Self::Q8),
            12 => Some(Self::Q4K),
            14 => Some(Self::Q6K),
            _ => None,
        }
    }

    /// Get bytes per element (0 for packed types)
    #[must_use]
    pub const fn bytes_per_element(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 | Self::I64 => 8,
            Self::I8 | Self::U8 | Self::Q8 => 1,
            Self::Q4 | Self::Q4K | Self::Q6K => 0, // Packed/block formats, need special handling
        }
    }

    /// Get type name
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F64 => "f64",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::Q4 => "q4",
            Self::Q8 => "q8",
            Self::Q4K => "q4_k",
            Self::Q6K => "q6_k",
        }
    }
}

// ============================================================================
// Alignment Utilities
// ============================================================================

/// Align value up to the nearest multiple of alignment
#[must_use]
pub const fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

/// Align value up to 64-byte boundary
#[must_use]
pub const fn align_64(value: usize) -> usize {
    align_up(value, ALIGNMENT)
}

/// Calculate padding needed to reach alignment
#[must_use]
pub const fn padding_to_align(value: usize, alignment: usize) -> usize {
    let aligned = align_up(value, alignment);
    aligned - value
}

/// Check if value is 64-byte aligned
#[must_use]
pub const fn is_aligned_64(value: usize) -> bool {
    value % ALIGNMENT == 0
}

// ============================================================================
// Writer
// ============================================================================

/// APR v2 format writer
#[derive(Debug)]
pub struct AprV2Writer {
    header: AprV2Header,
    metadata: AprV2Metadata,
    tensors: Vec<(TensorIndexEntry, Vec<u8>)>,
}

impl AprV2Writer {
    /// Create new writer
    #[must_use]
    pub fn new(metadata: AprV2Metadata) -> Self {
        Self {
            header: AprV2Header::new(),
            metadata,
            tensors: Vec::new(),
        }
    }

    /// Add tensor to the file
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        dtype: TensorDType,
        shape: Vec<usize>,
        data: Vec<u8>,
    ) {
        let entry = TensorIndexEntry::new(name, dtype, shape, 0, data.len() as u64);
        self.tensors.push((entry, data));
    }

    /// Add f32 tensor
    pub fn add_f32_tensor(&mut self, name: impl Into<String>, shape: Vec<usize>, data: &[f32]) {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.add_tensor(name, TensorDType::F32, shape, bytes);
    }

    /// Add f16 tensor (converts f32 → f16, 2 bytes per value)
    ///
    /// This provides true 2x compression over f32 storage with minimal precision loss
    /// for inference workloads. Uses IEEE 754 half-precision format.
    pub fn add_f16_tensor(&mut self, name: impl Into<String>, shape: Vec<usize>, data: &[f32]) {
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|&f| f32_to_f16(f).to_le_bytes())
            .collect();
        self.add_tensor(name, TensorDType::F16, shape, bytes);
    }

    /// Add Q8 tensor (8-bit symmetric quantization)
    ///
    /// Format: [scale: f32 (4 bytes)] + [quantized: i8 × n]
    /// Total size: 4 + n bytes (vs 4n for f32)
    /// Compression ratio: ~4x
    pub fn add_q8_tensor(&mut self, name: impl Into<String>, shape: Vec<usize>, data: &[f32]) {
        if data.is_empty() {
            self.add_tensor(name, TensorDType::Q8, shape, Vec::new());
            return;
        }

        // Find scale (max absolute value)
        let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };

        // Pack: scale (4 bytes) + quantized values (1 byte each)
        let mut bytes = Vec::with_capacity(4 + data.len());
        bytes.extend_from_slice(&scale.to_le_bytes());

        for &v in data {
            let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
            bytes.push(q as u8);
        }

        self.add_tensor(name, TensorDType::Q8, shape, bytes);
    }

    /// Add Q4 tensor (4-bit symmetric quantization, block-wise)
    ///
    /// Format: For each block of 32 values:
    ///   [block_scale: f16 (2 bytes)] + [packed nibbles: 16 bytes]
    ///
    /// Total size per block: 18 bytes (vs 128 bytes for f32)
    /// Compression ratio: ~7x
    pub fn add_q4_tensor(&mut self, name: impl Into<String>, shape: Vec<usize>, data: &[f32]) {
        const BLOCK_SIZE: usize = 32;

        if data.is_empty() {
            self.add_tensor(name, TensorDType::Q4, shape, Vec::new());
            return;
        }

        // Blocks: each block has 2-byte scale + 16 bytes of packed nibbles
        let num_blocks = (data.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let mut bytes = Vec::with_capacity(num_blocks * 18);

        for block_start in (0..data.len()).step_by(BLOCK_SIZE) {
            let block_end = (block_start + BLOCK_SIZE).min(data.len());
            let block = &data[block_start..block_end];

            // Find block scale
            let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };

            // Store scale as f16
            bytes.extend_from_slice(&f32_to_f16(scale).to_le_bytes());

            // Quantize and pack (2 values per byte)
            let mut packed_idx = 0;
            let mut packed_buf = [0u8; 16];

            for (i, &v) in block.iter().enumerate() {
                // Quantize to 4-bit signed (-8 to 7)
                let q = (v / scale).round().clamp(-8.0, 7.0) as i8;
                // Store as unsigned nibble (0-15)
                let nibble = ((q + 8) as u8) & 0x0F;

                if i % 2 == 0 {
                    packed_buf[packed_idx] = nibble;
                } else {
                    packed_buf[packed_idx] |= nibble << 4;
                    packed_idx += 1;
                }
            }
            // Note: No need to track packed_idx for odd elements since we write all 16 bytes anyway

            // Write all 16 bytes (zero-padded for partial blocks)
            bytes.extend_from_slice(&packed_buf);
        }

        self.add_tensor(name, TensorDType::Q4, shape, bytes);
    }

    /// Add raw Q4_K tensor (GGUF-compatible super-block format)
    ///
    /// This stores GGUF Q4_K data directly without re-quantization.
    /// Q4_K format: 256-element super-blocks with nested 32-element sub-blocks
    /// Each super-block: d (f16, 2B) + dmin (f16, 2B) + scales (12B) + qs (128B) = 144 bytes
    /// Effective bits per weight: ~4.5
    ///
    /// Use this when importing from GGUF to preserve exact quantization.
    pub fn add_q4k_raw_tensor(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        raw_data: Vec<u8>,
    ) {
        self.add_tensor(name, TensorDType::Q4K, shape, raw_data);
    }

    /// Add raw Q6_K tensor (GGUF-compatible super-block format)
    ///
    /// This stores GGUF Q6_K data directly without re-quantization.
    /// Q6_K format: 256-element super-blocks
    /// Each super-block: ql (128B) + qh (64B) + scales (16B) + d (f16, 2B) = 210 bytes
    /// Effective bits per weight: ~6.5
    pub fn add_q6k_raw_tensor(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        raw_data: Vec<u8>,
    ) {
        self.add_tensor(name, TensorDType::Q6K, shape, raw_data);
    }

    /// Set LZ4 compression flag
    pub fn with_lz4_compression(&mut self) -> &mut Self {
        self.header.flags = self.header.flags.with(AprV2Flags::LZ4_COMPRESSED);
        self
    }

    /// Set sharding info
    pub fn with_sharding(&mut self, shard_count: usize, shard_index: usize) -> &mut Self {
        self.header.flags = self.header.flags.with(AprV2Flags::SHARDED);
        self.metadata.sharding = Some(ShardingMetadata {
            shard_count,
            shard_index,
            total_size: 0,
            pattern: None,
        });
        self
    }

    /// Write to bytes
    ///
    /// # Errors
    /// Returns error if serialization fails.
    pub fn write(&mut self) -> Result<Vec<u8>, V2FormatError> {
        // Sort tensors by name
        self.tensors.sort_by(|a, b| a.0.name.cmp(&b.0.name));

        // Serialize metadata
        let metadata_bytes = self.metadata.to_json()?;
        let metadata_padded_size = align_64(metadata_bytes.len());

        // Build tensor index
        let mut tensor_index_bytes = Vec::new();
        let mut data_offset = 0_u64;

        for (entry, data) in &mut self.tensors {
            entry.offset = data_offset;
            entry.size = data.len() as u64;
            tensor_index_bytes.extend_from_slice(&entry.to_bytes());
            data_offset += align_64(data.len()) as u64;
        }
        let tensor_index_padded_size = align_64(tensor_index_bytes.len());

        // Calculate offsets
        let metadata_offset = HEADER_SIZE_V2;
        let tensor_index_offset = metadata_offset + metadata_padded_size;
        let data_section_offset = tensor_index_offset + tensor_index_padded_size;

        // Update header
        self.header.tensor_count = self.tensors.len() as u32;
        self.header.metadata_offset = metadata_offset as u64;
        self.header.metadata_size = metadata_bytes.len() as u32;
        self.header.tensor_index_offset = tensor_index_offset as u64;
        self.header.data_offset = data_section_offset as u64;
        self.header.update_checksum();

        // Build output
        let mut output = Vec::new();

        // Header
        output.extend_from_slice(&self.header.to_bytes());

        // Metadata (padded)
        output.extend_from_slice(&metadata_bytes);
        output.resize(metadata_offset + metadata_padded_size, 0);

        // Tensor index (padded)
        output.extend_from_slice(&tensor_index_bytes);
        output.resize(tensor_index_offset + tensor_index_padded_size, 0);

        // Tensor data (each 64-byte aligned)
        for (_, data) in &self.tensors {
            let start = output.len();
            output.extend_from_slice(data);
            let padded_size = align_64(data.len());
            output.resize(start + padded_size, 0);
        }

        // Footer checksum
        let footer_checksum = crc32(&output);
        output.extend_from_slice(&footer_checksum.to_le_bytes());

        Ok(output)
    }

    /// Write to a Write impl
    ///
    /// # Errors
    /// Returns error if write fails.
    pub fn write_to<W: Write>(&mut self, writer: &mut W) -> Result<(), V2FormatError> {
        let bytes = self.write()?;
        writer
            .write_all(&bytes)
            .map_err(|e| V2FormatError::IoError(e.to_string()))
    }
}

// ============================================================================
// Reader
// ============================================================================

/// APR v2 format reader (owns data - copies input)
#[derive(Debug)]
pub struct AprV2Reader {
    header: AprV2Header,
    metadata: AprV2Metadata,
    tensor_index: Vec<TensorIndexEntry>,
    data: Vec<u8>,
}

/// APR v2 format reader with zero-copy (borrows data - for mmap)
///
/// This reader borrows the data slice instead of copying it, enabling
/// true zero-copy access when used with memory-mapped files.
///
/// # Example
///
/// ```ignore
/// use aprender::bundle::MappedFile;
/// use aprender::format::v2::AprV2ReaderRef;
///
/// let mmap = MappedFile::open("model.apr")?;
/// let reader = AprV2ReaderRef::from_bytes(mmap.as_slice())?;
/// let weights = reader.get_f32_tensor("embed_tokens.weight")?;
/// ```
#[derive(Debug)]
pub struct AprV2ReaderRef<'a> {
    header: AprV2Header,
    metadata: AprV2Metadata,
    tensor_index: Vec<TensorIndexEntry>,
    data: &'a [u8],
}

impl AprV2Reader {
    /// Read from bytes
    ///
    /// # Errors
    /// Returns error if parsing fails.
    pub fn from_bytes(data: &[u8]) -> Result<Self, V2FormatError> {
        if data.len() < HEADER_SIZE_V2 {
            return Err(V2FormatError::InvalidHeader("file too small".to_string()));
        }

        // Parse header
        let header = AprV2Header::from_bytes(data)?;

        // Verify checksum
        if !header.verify_checksum() {
            return Err(V2FormatError::ChecksumMismatch);
        }

        // Parse metadata
        let metadata_start = header.metadata_offset as usize;
        let metadata_end = metadata_start + header.metadata_size as usize;

        if data.len() < metadata_end {
            return Err(V2FormatError::InvalidHeader(
                "file too small for metadata".to_string(),
            ));
        }

        let metadata = AprV2Metadata::from_json(&data[metadata_start..metadata_end])?;

        // Parse tensor index
        let index_start = header.tensor_index_offset as usize;
        let mut tensor_index = Vec::with_capacity(header.tensor_count as usize);
        let mut pos = index_start;

        for _ in 0..header.tensor_count {
            let (entry, consumed) = TensorIndexEntry::from_bytes(&data[pos..])?;
            tensor_index.push(entry);
            pos += consumed;
        }

        // Verify tensor names are sorted
        for i in 1..tensor_index.len() {
            if tensor_index[i].name < tensor_index[i - 1].name {
                return Err(V2FormatError::InvalidTensorIndex(
                    "tensor index not sorted".to_string(),
                ));
            }
        }

        Ok(Self {
            header,
            metadata,
            tensor_index,
            data: data.to_vec(),
        })
    }

    /// Read from a Read impl
    ///
    /// # Errors
    /// Returns error if read fails.
    pub fn from_reader<R: Read>(reader: &mut R) -> Result<Self, V2FormatError> {
        let mut data = Vec::new();
        reader
            .read_to_end(&mut data)
            .map_err(|e| V2FormatError::IoError(e.to_string()))?;
        Self::from_bytes(&data)
    }

    /// Get header
    #[must_use]
    pub fn header(&self) -> &AprV2Header {
        &self.header
    }

    /// Get metadata
    #[must_use]
    pub fn metadata(&self) -> &AprV2Metadata {
        &self.metadata
    }

    /// Get tensor names
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_index.iter().map(|e| e.name.as_str()).collect()
    }

    /// Get tensor by name
    #[must_use]
    pub fn get_tensor(&self, name: &str) -> Option<&TensorIndexEntry> {
        self.tensor_index.iter().find(|e| e.name == name)
    }

    /// Get tensor data by name
    #[must_use]
    pub fn get_tensor_data(&self, name: &str) -> Option<&[u8]> {
        let entry = self.get_tensor(name)?;
        let start = (self.header.data_offset + entry.offset) as usize;
        let end = start + entry.size as usize;

        if end <= self.data.len() {
            Some(&self.data[start..end])
        } else {
            None
        }
    }

    /// Get tensor as f32 slice (F32 dtype only)
    #[must_use]
    pub fn get_f32_tensor(&self, name: &str) -> Option<Vec<f32>> {
        let entry = self.get_tensor(name)?;
        if entry.dtype != TensorDType::F32 {
            return None;
        }

        let data = self.get_tensor_data(name)?;
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Some(floats)
    }

    /// Get tensor as f32 Vec, dequantizing if necessary
    ///
    /// Supports all tensor types:
    /// - F32: direct copy
    /// - F16: IEEE 754 half-precision → f32
    /// - Q8: 8-bit symmetric dequantization
    /// - Q4: 4-bit block dequantization
    /// - Q4K: GGUF Q4_K super-block dequantization (GH-200)
    /// - Q6K: GGUF Q6_K super-block dequantization (GH-200)
    #[must_use]
    pub fn get_tensor_as_f32(&self, name: &str) -> Option<Vec<f32>> {
        let entry = self.get_tensor(name)?;
        let data = self.get_tensor_data(name)?;
        let element_count = entry.element_count();

        match entry.dtype {
            TensorDType::F32 => {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Some(floats)
            }
            TensorDType::F16 => {
                let floats: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                    .collect();
                Some(floats)
            }
            TensorDType::Q8 => {
                if data.len() < 4 {
                    return None;
                }
                let scale = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                let floats: Vec<f32> = data[4..]
                    .iter()
                    .map(|&b| f32::from(b as i8) * scale)
                    .collect();
                Some(floats)
            }
            TensorDType::Q4 => Some(dequantize_q4(data, element_count)),
            TensorDType::Q4K => dequantize_q4_k(data, 0, element_count).ok(),
            TensorDType::Q6K => dequantize_q6_k(data, 0, element_count).ok(),
            _ => None, // Other types not yet supported
        }
    }

    /// Check if all tensors are 64-byte aligned
    #[must_use]
    pub fn verify_alignment(&self) -> bool {
        let data_offset = self.header.data_offset as usize;
        self.tensor_index
            .iter()
            .all(|e| is_aligned_64(data_offset + e.offset as usize))
    }
}

impl<'a> AprV2ReaderRef<'a> {
    /// Read from bytes (zero-copy - borrows data)
    ///
    /// Unlike `AprV2Reader::from_bytes`, this does NOT copy the input data.
    /// The reader borrows the slice, making it ideal for use with mmap.
    ///
    /// # Errors
    /// Returns error if parsing fails.
    pub fn from_bytes(data: &'a [u8]) -> Result<Self, V2FormatError> {
        if data.len() < HEADER_SIZE_V2 {
            return Err(V2FormatError::InvalidHeader("file too small".to_string()));
        }

        // Parse header
        let header = AprV2Header::from_bytes(data)?;

        // Verify checksum
        if !header.verify_checksum() {
            return Err(V2FormatError::ChecksumMismatch);
        }

        // Parse metadata
        let metadata_start = header.metadata_offset as usize;
        let metadata_end = metadata_start + header.metadata_size as usize;

        if data.len() < metadata_end {
            return Err(V2FormatError::InvalidHeader(
                "file too small for metadata".to_string(),
            ));
        }

        let metadata = AprV2Metadata::from_json(&data[metadata_start..metadata_end])?;

        // Parse tensor index
        let index_start = header.tensor_index_offset as usize;
        let mut tensor_index = Vec::with_capacity(header.tensor_count as usize);
        let mut pos = index_start;

        for _ in 0..header.tensor_count {
            let (entry, consumed) = TensorIndexEntry::from_bytes(&data[pos..])?;
            tensor_index.push(entry);
            pos += consumed;
        }

        // Verify tensor names are sorted
        for i in 1..tensor_index.len() {
            if tensor_index[i].name < tensor_index[i - 1].name {
                return Err(V2FormatError::InvalidTensorIndex(
                    "tensor index not sorted".to_string(),
                ));
            }
        }

        Ok(Self {
            header,
            metadata,
            tensor_index,
            data, // Borrow, no copy!
        })
    }

    /// Get header
    #[must_use]
    pub fn header(&self) -> &AprV2Header {
        &self.header
    }

    /// Get metadata
    #[must_use]
    pub fn metadata(&self) -> &AprV2Metadata {
        &self.metadata
    }

    /// Get tensor names
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_index.iter().map(|e| e.name.as_str()).collect()
    }

    /// Get tensor by name
    #[must_use]
    pub fn get_tensor(&self, name: &str) -> Option<&TensorIndexEntry> {
        self.tensor_index.iter().find(|e| e.name == name)
    }

    /// Get tensor data by name (zero-copy slice into mmap)
    #[must_use]
    pub fn get_tensor_data(&self, name: &str) -> Option<&[u8]> {
        let entry = self.get_tensor(name)?;
        let start = (self.header.data_offset + entry.offset) as usize;
        let end = start + entry.size as usize;

        if end <= self.data.len() {
            Some(&self.data[start..end])
        } else {
            None
        }
    }

    /// Get tensor as f32 Vec (copies data from mmap to `Vec<f32>`)
    ///
    /// Note: This allocates memory for the f32 values. For very large tensors,
    /// consider using `get_tensor_data` and processing in chunks.
    #[must_use]
    pub fn get_f32_tensor(&self, name: &str) -> Option<Vec<f32>> {
        let entry = self.get_tensor(name)?;
        if entry.dtype != TensorDType::F32 {
            return None;
        }

        let data = self.get_tensor_data(name)?;
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Some(floats)
    }

    /// Get tensor as f32 Vec, dequantizing if necessary
    ///
    /// Supports all tensor types:
    /// - F32: direct copy
    /// - F16: IEEE 754 half-precision → f32
    /// - Q8: 8-bit symmetric dequantization
    /// - Q4: 4-bit block dequantization
    /// - Q4K: GGUF Q4_K super-block dequantization (GH-200)
    /// - Q6K: GGUF Q6_K super-block dequantization (GH-200)
    #[must_use]
    pub fn get_tensor_as_f32(&self, name: &str) -> Option<Vec<f32>> {
        let entry = self.get_tensor(name)?;
        let data = self.get_tensor_data(name)?;
        let element_count = entry.element_count();

        match entry.dtype {
            TensorDType::F32 => {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Some(floats)
            }
            TensorDType::F16 => {
                let floats: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                    .collect();
                Some(floats)
            }
            TensorDType::Q8 => {
                if data.len() < 4 {
                    return None;
                }
                let scale = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                let floats: Vec<f32> = data[4..]
                    .iter()
                    .map(|&b| f32::from(b as i8) * scale)
                    .collect();
                Some(floats)
            }
            TensorDType::Q4 => Some(dequantize_q4(data, element_count)),
            TensorDType::Q4K => dequantize_q4_k(data, 0, element_count).ok(),
            TensorDType::Q6K => dequantize_q6_k(data, 0, element_count).ok(),
            _ => None, // Other types not yet supported
        }
    }

    /// Check if all tensors are 64-byte aligned
    #[must_use]
    pub fn verify_alignment(&self) -> bool {
        let data_offset = self.header.data_offset as usize;
        self.tensor_index
            .iter()
            .all(|e| is_aligned_64(data_offset + e.offset as usize))
    }
}

// ============================================================================
// Shard Manifest
// ============================================================================

/// Shard manifest for multi-file models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardManifest {
    /// Format version
    pub version: String,
    /// Total number of shards
    pub shard_count: usize,
    /// Total size in bytes
    pub total_size: u64,
    /// Total tensor count
    pub tensor_count: usize,
    /// Shard files
    pub shards: Vec<ShardInfo>,
    /// Tensor to shard mapping
    pub weight_map: HashMap<String, usize>,
}

/// Information about a single shard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    /// Shard filename
    pub filename: String,
    /// Shard index
    pub index: usize,
    /// Size in bytes
    pub size: u64,
    /// Tensor names in this shard
    pub tensors: Vec<String>,
}

impl ShardManifest {
    /// Create new empty manifest
    #[must_use]
    pub fn new(shard_count: usize) -> Self {
        Self {
            version: "2.0".to_string(),
            shard_count,
            total_size: 0,
            tensor_count: 0,
            shards: Vec::with_capacity(shard_count),
            weight_map: HashMap::new(),
        }
    }

    /// Add shard info
    pub fn add_shard(&mut self, info: ShardInfo) {
        for tensor in &info.tensors {
            self.weight_map.insert(tensor.clone(), info.index);
        }
        self.tensor_count += info.tensors.len();
        self.total_size += info.size;
        self.shards.push(info);
    }

    /// Get shard index for tensor
    #[must_use]
    pub fn shard_for_tensor(&self, name: &str) -> Option<usize> {
        self.weight_map.get(name).copied()
    }

    /// Serialize to JSON
    ///
    /// # Errors
    /// Returns error if serialization fails.
    pub fn to_json(&self) -> Result<String, V2FormatError> {
        serde_json::to_string_pretty(self).map_err(|e| V2FormatError::MetadataError(e.to_string()))
    }

    /// Deserialize from JSON
    ///
    /// # Errors
    /// Returns error if deserialization fails.
    pub fn from_json(json: &str) -> Result<Self, V2FormatError> {
        serde_json::from_str(json).map_err(|e| V2FormatError::MetadataError(e.to_string()))
    }
}

// ============================================================================
// Error Type
// ============================================================================

/// APR v2 format error
#[derive(Debug, Clone, PartialEq)]
pub enum V2FormatError {
    /// Invalid magic number
    InvalidMagic([u8; 4]),
    /// Invalid header
    InvalidHeader(String),
    /// Invalid tensor index
    InvalidTensorIndex(String),
    /// Metadata error
    MetadataError(String),
    /// Checksum mismatch
    ChecksumMismatch,
    /// Alignment error
    AlignmentError(String),
    /// I/O error
    IoError(String),
    /// Compression error
    CompressionError(String),
}

impl std::fmt::Display for V2FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMagic(magic) => {
                write!(
                    f,
                    "Invalid magic: {:02x}{:02x}{:02x}{:02x}",
                    magic[0], magic[1], magic[2], magic[3]
                )
            }
            Self::InvalidHeader(msg) => write!(f, "Invalid header: {msg}"),
            Self::InvalidTensorIndex(msg) => write!(f, "Invalid tensor index: {msg}"),
            Self::MetadataError(msg) => write!(f, "Metadata error: {msg}"),
            Self::ChecksumMismatch => write!(f, "Checksum mismatch"),
            Self::AlignmentError(msg) => write!(f, "Alignment error: {msg}"),
            Self::IoError(msg) => write!(f, "I/O error: {msg}"),
            Self::CompressionError(msg) => write!(f, "Compression error: {msg}"),
        }
    }
}

impl std::error::Error for V2FormatError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
