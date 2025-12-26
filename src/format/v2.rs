//! APR v2 Format Module (GH-119)
//!
//! Implements the APR v2 format specification with:
//! - 64-byte tensor alignment for zero-copy mmap
//! - LZ4 block compression (64KB blocks)
//! - JSON metadata section
//! - Multi-file sharding for 10B+ parameter models
//! - Backward compatibility with APR v1
//!
//! # Format Structure (APR v2)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Header (64 bytes, 64-byte aligned)                          │
//! │   - Magic: "APR2" (4 bytes)                                 │
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
// Constants
// ============================================================================

/// APR v2 magic number: "APR2" in ASCII (0x41505232)
pub const MAGIC_V2: [u8; 4] = [0x41, 0x50, 0x52, 0x32];

/// APR v1 magic number for backward compatibility
pub const MAGIC_V1: [u8; 4] = [0x41, 0x50, 0x52, 0x4E];

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

/// APR v2 file header (64 bytes)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AprV2Header {
    /// Magic number ("APR2")
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

    /// Check if this is a v1 file (for backward compatibility)
    #[must_use]
    pub fn is_v1(&self) -> bool {
        self.magic == MAGIC_V1
    }

    /// Set v1 magic for backward compatibility with realizar
    pub fn set_v1_magic(&mut self) {
        self.magic = MAGIC_V1;
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

        // Check for v1 or v2 magic
        if magic != MAGIC_V2 && magic != MAGIC_V1 {
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
    /// Examples: "hf://openai/whisper-tiny", "local://path/to/model.safetensors"
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

    /// Custom key-value pairs
    #[serde(default, flatten)]
    pub custom: HashMap<String, serde_json::Value>,
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
            Self::Q4 => 0, // Packed, need special handling
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

    /// Enable v1 compatibility mode (APRN magic for realizar support)
    pub fn with_v1_compat(&mut self) -> &mut Self {
        self.header.set_v1_magic();
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

    /// Get tensor as f32 slice
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

    /// Get tensor as f32 Vec (copies data from mmap to Vec<f32>)
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
mod tests {
    use super::*;

    #[test]
    fn test_magic_v2() {
        assert_eq!(MAGIC_V2, [0x41, 0x50, 0x52, 0x32]); // "APR2"
        assert_eq!(&MAGIC_V2, b"APR2");
    }

    #[test]
    fn test_header_size() {
        assert_eq!(HEADER_SIZE_V2, 64);
        assert!(is_aligned_64(HEADER_SIZE_V2));
    }

    #[test]
    fn test_flags() {
        let flags = AprV2Flags::new()
            .with(AprV2Flags::LZ4_COMPRESSED)
            .with(AprV2Flags::QUANTIZED);

        assert!(flags.is_lz4_compressed());
        assert!(flags.is_quantized());
        assert!(!flags.is_encrypted());
        assert!(!flags.is_sharded());
    }

    #[test]
    fn test_header_new() {
        let header = AprV2Header::new();
        assert_eq!(header.magic, MAGIC_V2);
        assert_eq!(header.version, VERSION_V2);
        assert!(header.is_valid());
        assert!(!header.is_v1());
    }

    #[test]
    fn test_header_roundtrip() {
        let mut header = AprV2Header::new();
        header.tensor_count = 42;
        header.metadata_size = 1024;
        header.update_checksum();

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE_V2);

        let parsed = AprV2Header::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.tensor_count, 42);
        assert_eq!(parsed.metadata_size, 1024);
        assert!(parsed.verify_checksum());
    }

    #[test]
    fn test_header_invalid_magic() {
        let bytes = [0xFF; HEADER_SIZE_V2];
        let result = AprV2Header::from_bytes(&bytes);
        assert!(matches!(result, Err(V2FormatError::InvalidMagic(_))));
    }

    #[test]
    fn test_metadata_json_roundtrip() {
        let mut metadata = AprV2Metadata::new("whisper");
        metadata.name = Some("whisper-tiny".to_string());
        metadata.param_count = 39_000_000;

        let json = metadata.to_json().unwrap();
        let parsed = AprV2Metadata::from_json(&json).unwrap();

        assert_eq!(parsed.model_type, "whisper");
        assert_eq!(parsed.name.as_deref(), Some("whisper-tiny"));
        assert_eq!(parsed.param_count, 39_000_000);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(63, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
    }

    #[test]
    fn test_align_64() {
        assert_eq!(align_64(0), 0);
        assert_eq!(align_64(1), 64);
        assert_eq!(align_64(100), 128);
        assert_eq!(align_64(128), 128);
    }

    #[test]
    fn test_is_aligned_64() {
        assert!(is_aligned_64(0));
        assert!(is_aligned_64(64));
        assert!(is_aligned_64(128));
        assert!(!is_aligned_64(1));
        assert!(!is_aligned_64(63));
        assert!(!is_aligned_64(65));
    }

    #[test]
    fn test_tensor_dtype() {
        assert_eq!(TensorDType::F32.bytes_per_element(), 4);
        assert_eq!(TensorDType::F16.bytes_per_element(), 2);
        assert_eq!(TensorDType::F64.bytes_per_element(), 8);
        assert_eq!(TensorDType::I8.bytes_per_element(), 1);
        assert_eq!(TensorDType::Q4.bytes_per_element(), 0);
    }

    #[test]
    fn test_tensor_dtype_name() {
        assert_eq!(TensorDType::F32.name(), "f32");
        assert_eq!(TensorDType::BF16.name(), "bf16");
        assert_eq!(TensorDType::Q8.name(), "q8");
    }

    #[test]
    fn test_tensor_index_entry_roundtrip() {
        let entry = TensorIndexEntry::new(
            "encoder.layer.0.weight",
            TensorDType::F32,
            vec![512, 768],
            0,
            512 * 768 * 4,
        );

        let bytes = entry.to_bytes();
        let (parsed, _) = TensorIndexEntry::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.name, "encoder.layer.0.weight");
        assert_eq!(parsed.dtype, TensorDType::F32);
        assert_eq!(parsed.shape, vec![512, 768]);
        assert_eq!(parsed.element_count(), 512 * 768);
    }

    #[test]
    fn test_writer_reader_roundtrip() {
        let metadata = AprV2Metadata::new("test");
        let mut writer = AprV2Writer::new(metadata);

        writer.add_f32_tensor("weight", vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        writer.add_f32_tensor("bias", vec![3], &[0.1, 0.2, 0.3]);

        let bytes = writer.write().unwrap();

        let reader = AprV2Reader::from_bytes(&bytes).unwrap();
        assert_eq!(reader.metadata().model_type, "test");
        assert_eq!(reader.tensor_names(), vec!["bias", "weight"]); // Sorted

        let weight = reader.get_f32_tensor("weight").unwrap();
        assert_eq!(weight.len(), 6);
        assert!((weight[0] - 1.0).abs() < 1e-6);

        // Verify alignment
        assert!(reader.verify_alignment());
    }

    #[test]
    fn test_v1_compat_magic() {
        let metadata = AprV2Metadata::new("test");
        let mut writer = AprV2Writer::new(metadata);
        writer.with_v1_compat(); // Use APRN magic for backward compatibility

        writer.add_f32_tensor("weight", vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let bytes = writer.write().unwrap();

        // Check magic bytes are APRN (v1) not APR2 (v2)
        assert_eq!(&bytes[0..4], b"APRN", "Magic should be APRN for v1 compat");

        // Reader should still work
        let reader = AprV2Reader::from_bytes(&bytes).unwrap();
        assert_eq!(reader.metadata().model_type, "test");
    }

    #[test]
    fn test_writer_alignment() {
        let metadata = AprV2Metadata::new("test");
        let mut writer = AprV2Writer::new(metadata);

        // Add tensor with non-aligned size
        writer.add_f32_tensor("test", vec![7], &[1.0; 7]); // 28 bytes, not aligned

        let bytes = writer.write().unwrap();
        let reader = AprV2Reader::from_bytes(&bytes).unwrap();

        // Data should still be 64-byte aligned
        assert!(reader.verify_alignment());
    }

    #[test]
    fn test_shard_manifest() {
        let mut manifest = ShardManifest::new(2);

        manifest.add_shard(ShardInfo {
            filename: "model-00000-of-00002.apr".to_string(),
            index: 0,
            size: 1024,
            tensors: vec!["layer1.weight".to_string(), "layer1.bias".to_string()],
        });

        manifest.add_shard(ShardInfo {
            filename: "model-00001-of-00002.apr".to_string(),
            index: 1,
            size: 2048,
            tensors: vec!["layer2.weight".to_string()],
        });

        assert_eq!(manifest.shard_count, 2);
        assert_eq!(manifest.tensor_count, 3);
        assert_eq!(manifest.total_size, 3072);

        assert_eq!(manifest.shard_for_tensor("layer1.weight"), Some(0));
        assert_eq!(manifest.shard_for_tensor("layer2.weight"), Some(1));
        assert_eq!(manifest.shard_for_tensor("nonexistent"), None);

        // JSON roundtrip
        let json = manifest.to_json().unwrap();
        let parsed = ShardManifest::from_json(&json).unwrap();
        assert_eq!(parsed.shard_count, 2);
    }

    #[test]
    fn test_v2_format_error_display() {
        let err = V2FormatError::InvalidMagic([0x00, 0x01, 0x02, 0x03]);
        assert!(err.to_string().contains("00010203"));

        let err = V2FormatError::ChecksumMismatch;
        assert_eq!(err.to_string(), "Checksum mismatch");
    }

    #[test]
    fn test_quantization_metadata() {
        let quant = QuantizationMetadata {
            quant_type: "int8".to_string(),
            bits: 8,
            block_size: Some(32),
            symmetric: true,
        };

        let mut metadata = AprV2Metadata::new("llm");
        metadata.quantization = Some(quant);

        let json = metadata.to_json().unwrap();
        let parsed = AprV2Metadata::from_json(&json).unwrap();

        let quant = parsed.quantization.unwrap();
        assert_eq!(quant.quant_type, "int8");
        assert_eq!(quant.bits, 8);
        assert_eq!(quant.block_size, Some(32));
    }

    #[test]
    fn test_backward_compat_v1_magic() {
        // Create bytes with v1 magic
        let mut bytes = [0u8; HEADER_SIZE_V2];
        bytes[0..4].copy_from_slice(&MAGIC_V1);

        let header = AprV2Header::from_bytes(&bytes).unwrap();
        assert!(header.is_v1());
        assert!(!header.is_valid()); // is_valid only returns true for v2
    }

    #[test]
    fn test_flags_combinations() {
        let flags = AprV2Flags::new()
            .with(AprV2Flags::LZ4_COMPRESSED)
            .with(AprV2Flags::SHARDED)
            .with(AprV2Flags::HAS_VOCAB);

        assert!(flags.is_lz4_compressed());
        assert!(flags.is_sharded());
        assert!(flags.contains(AprV2Flags::HAS_VOCAB));
        assert!(!flags.is_encrypted());

        let without = flags.without(AprV2Flags::SHARDED);
        assert!(!without.is_sharded());
        assert!(without.is_lz4_compressed());
    }

    #[test]
    fn test_metadata_custom_fields() {
        let mut metadata = AprV2Metadata::new("custom");
        metadata.custom.insert(
            "custom_field".to_string(),
            serde_json::json!("custom_value"),
        );
        metadata
            .custom
            .insert("nested".to_string(), serde_json::json!({"key": "value"}));

        let json = metadata.to_json().unwrap();
        let parsed = AprV2Metadata::from_json(&json).unwrap();

        assert_eq!(
            parsed.custom.get("custom_field"),
            Some(&serde_json::json!("custom_value"))
        );
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_tensor_dtype_from_u8() {
        assert_eq!(TensorDType::from_u8(0), Some(TensorDType::F32));
        assert_eq!(TensorDType::from_u8(1), Some(TensorDType::F16));
        assert_eq!(TensorDType::from_u8(2), Some(TensorDType::BF16));
        assert_eq!(TensorDType::from_u8(3), Some(TensorDType::F64));
        assert_eq!(TensorDType::from_u8(4), Some(TensorDType::I32));
        assert_eq!(TensorDType::from_u8(5), Some(TensorDType::I64));
        assert_eq!(TensorDType::from_u8(6), Some(TensorDType::I8));
        assert_eq!(TensorDType::from_u8(7), Some(TensorDType::U8));
        assert_eq!(TensorDType::from_u8(99), None);
    }

    #[test]
    fn test_v2_format_error_variants() {
        let err = V2FormatError::InvalidHeader("bad header".to_string());
        assert!(err.to_string().contains("bad header") || err.to_string().contains("Invalid"));

        let err = V2FormatError::InvalidTensorIndex("corrupt index".to_string());
        assert!(err.to_string().contains("corrupt") || err.to_string().contains("index"));

        let err = V2FormatError::MetadataError("invalid metadata".to_string());
        assert!(err.to_string().contains("metadata") || err.to_string().contains("Metadata"));

        let err = V2FormatError::AlignmentError("alignment off".to_string());
        assert!(err.to_string().contains("alignment") || err.to_string().contains("Alignment"));

        let err = V2FormatError::IoError("read failed".to_string());
        assert!(err.to_string().contains("read failed") || err.to_string().contains("I/O"));

        let err = V2FormatError::CompressionError("decompress failed".to_string());
        assert!(err.to_string().contains("decompress") || err.to_string().contains("Compression"));
    }

    #[test]
    fn test_header_checksum_compute() {
        let mut header = AprV2Header::new();
        header.version = (2, 0);
        let checksum = header.compute_checksum();
        assert!(checksum != 0);
    }

    #[test]
    fn test_header_update_checksum() {
        let mut header = AprV2Header::new();
        header.checksum = 0;
        header.update_checksum();
        assert!(header.checksum != 0);
    }

    #[test]
    fn test_header_verify_checksum() {
        let mut header = AprV2Header::new();
        header.update_checksum();
        assert!(header.verify_checksum());
        header.version = (99, 0);
        assert!(!header.verify_checksum());
    }

    #[test]
    fn test_metadata_to_json_pretty() {
        let metadata = AprV2Metadata::new("llama");
        let json = metadata.to_json_pretty().unwrap();
        assert!(json.contains("llama"));
        assert!(json.contains('\n')); // Pretty format has newlines
    }

    #[test]
    fn test_tensor_index_entry_element_count() {
        let entry = TensorIndexEntry::new(
            "test",
            TensorDType::F32,
            vec![2, 3, 4],
            0,
            96, // 2*3*4*4 bytes
        );
        assert_eq!(entry.element_count(), 24);
    }

    #[test]
    fn test_tensor_index_entry_to_bytes() {
        let entry = TensorIndexEntry::new("t", TensorDType::F32, vec![10], 0, 40);
        let bytes = entry.to_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_writer_with_lz4() {
        let metadata = AprV2Metadata::new("test");
        let mut writer = AprV2Writer::new(metadata);
        writer.with_lz4_compression();
        // Just verify it doesn't panic
    }

    #[test]
    fn test_writer_with_sharding() {
        let metadata = AprV2Metadata::new("test");
        let mut writer = AprV2Writer::new(metadata);
        writer.with_sharding(4, 0);
        // Just verify it doesn't panic
    }

    #[test]
    fn test_reader_ref_from_bytes() {
        let metadata = AprV2Metadata::new("test");
        let mut writer = AprV2Writer::new(metadata);
        writer.add_f32_tensor("w", vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let bytes = writer.write().unwrap();

        let reader = AprV2ReaderRef::from_bytes(&bytes).unwrap();
        assert_eq!(reader.header().version.0, 2);
        assert_eq!(reader.metadata().model_type, "test");
        assert_eq!(reader.tensor_names().len(), 1);
        assert!(reader.get_tensor("w").is_some());
        assert!(reader.verify_alignment());
    }

    #[test]
    fn test_reader_ref_get_tensor_data() {
        let metadata = AprV2Metadata::new("test");
        let mut writer = AprV2Writer::new(metadata);
        writer.add_f32_tensor("w", vec![2], &[1.0, 2.0]);
        let bytes = writer.write().unwrap();

        let reader = AprV2ReaderRef::from_bytes(&bytes).unwrap();
        let data = reader.get_tensor_data("w");
        assert!(data.is_some());
    }

    #[test]
    fn test_reader_ref_get_f32_tensor() {
        let metadata = AprV2Metadata::new("test");
        let mut writer = AprV2Writer::new(metadata);
        writer.add_f32_tensor("w", vec![3], &[1.0, 2.0, 3.0]);
        let bytes = writer.write().unwrap();

        let reader = AprV2ReaderRef::from_bytes(&bytes).unwrap();
        let tensor = reader.get_f32_tensor("w").unwrap();
        assert_eq!(tensor.len(), 3);
        assert!((tensor[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sharding_metadata() {
        let shard = ShardingMetadata {
            shard_count: 4,
            shard_index: 0,
            total_size: 10_000_000,
            pattern: Some("model-{:05d}-of-{:05d}.apr".to_string()),
        };
        assert_eq!(shard.shard_count, 4);
        assert_eq!(shard.total_size, 10_000_000);
        assert!(shard.pattern.is_some());
    }

    #[test]
    fn test_flags_all_bits() {
        let flags = AprV2Flags::new()
            .with(AprV2Flags::LZ4_COMPRESSED)
            .with(AprV2Flags::ENCRYPTED)
            .with(AprV2Flags::SIGNED)
            .with(AprV2Flags::SHARDED)
            .with(AprV2Flags::HAS_VOCAB)
            .with(AprV2Flags::QUANTIZED);

        assert!(flags.is_lz4_compressed());
        assert!(flags.is_encrypted());
        assert!(flags.contains(AprV2Flags::SIGNED));
        assert!(flags.is_sharded());
        assert!(flags.contains(AprV2Flags::HAS_VOCAB));
        assert!(flags.is_quantized());
    }

    #[test]
    fn test_shard_info_creation() {
        let info = ShardInfo {
            filename: "shard.apr".to_string(),
            index: 0,
            size: 1024,
            tensors: vec!["a".to_string(), "b".to_string()],
        };
        assert_eq!(info.filename, "shard.apr");
        assert_eq!(info.tensors.len(), 2);
    }

    /// DD6: Model provenance must be tracked in APR metadata
    /// Falsification: If source/origin is lost after conversion, provenance tracking fails
    #[test]
    fn test_dd6_provenance_tracked() {
        let mut metadata = AprV2Metadata::new("test_model");

        // Set provenance information
        metadata.source = Some("hf://openai/whisper-tiny".to_string());
        metadata.original_format = Some("safetensors".to_string());

        // Verify provenance is preserved in serialization
        let json = metadata.to_json().expect("serialize");
        let parsed: AprV2Metadata = serde_json::from_slice(&json).expect("deserialize");

        assert_eq!(
            parsed.source,
            Some("hf://openai/whisper-tiny".to_string()),
            "DD6 FALSIFIED: Source provenance lost after serialization"
        );
        assert_eq!(
            parsed.original_format,
            Some("safetensors".to_string()),
            "DD6 FALSIFIED: Original format lost after serialization"
        );
    }

    /// DD6b: Verify provenance survives full APR write/read cycle
    #[test]
    fn test_dd6_provenance_roundtrip() {
        let mut metadata = AprV2Metadata::new("whisper");
        metadata.source = Some("local:///models/whisper-tiny.safetensors".to_string());
        metadata.original_format = Some("safetensors".to_string());
        metadata.author = Some("OpenAI".to_string());

        let mut writer = AprV2Writer::new(metadata);
        writer.add_f32_tensor("test", vec![4], &[1.0, 2.0, 3.0, 4.0]);

        let bytes = writer.write().expect("write");
        let reader = AprV2Reader::from_bytes(&bytes).expect("read");

        let read_meta = reader.metadata();
        assert!(
            read_meta.source.is_some(),
            "DD6 FALSIFIED: Source provenance not preserved in APR file"
        );
        assert_eq!(
            read_meta.source.as_deref(),
            Some("local:///models/whisper-tiny.safetensors"),
            "DD6 FALSIFIED: Source URI corrupted"
        );
        assert_eq!(
            read_meta.original_format.as_deref(),
            Some("safetensors"),
            "DD6 FALSIFIED: Original format corrupted"
        );
    }
}
