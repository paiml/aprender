//! Aprender .apr format support for realizar (APR v2 only)
//!
//! This module provides loading and inference for models in Aprender's native
//! .apr v2 format (Magic: `APR\0` = 0x41505232).
//!
//! ## Format Structure (APR v2, 64-byte header)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Header (64 bytes)                                           │
//! │   - Magic: "APR\0" (4 bytes)                                 │
//! │   - Version: major.minor (2 bytes)                          │
//! │   - Flags (2 bytes)                                         │
//! │   - Tensor count (4 bytes)                                  │
//! │   - Metadata offset (8 bytes)                               │
//! │   - Metadata size (4 bytes)                                 │
//! │   - Tensor index offset (8 bytes)                           │
//! │   - Data offset (8 bytes)                                   │
//! │   - Checksum (4 bytes)                                      │
//! │   - Reserved (20 bytes)                                     │
//! ├─────────────────────────────────────────────────────────────┤
//! │ JSON Metadata (padded to 64-byte boundary)                  │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Tensor Index (sorted by name)                               │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Tensor Data (each tensor 64-byte aligned)                   │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::apr::AprV2Model;
//!
//! let model = AprV2Model::load("model.apr")?;
//! println!("Tensors: {}", model.tensor_count());
//! ```

use std::collections::HashMap;
use std::fs::{self, File};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{RealizarError, Result};
use crate::safetensors::find_sibling_file;
use crate::tokenizer::SentencePieceTokenizer;

// PMAT-802: Extracted modules
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests;
pub mod dequant;
mod helpers;
mod model_data;
pub(crate) mod tokenizer;

// PMAT-COMPLY: Re-export ModelData
pub use model_data::ModelData;

// PMAT-COMPLY: Re-export dequantization functions
pub(crate) use dequant::{dequantize_f16, dequantize_q4_k, dequantize_q6_k, dequantize_q8_0};
// Re-export for tests
pub use dequant::{dtype_to_ggml_qtype, f16_to_f32, is_quantized_dtype};

// T-COV-95: Test factory for Active Pygmy APR models
#[cfg(test)]
pub(crate) mod test_factory;

#[cfg(feature = "cuda")]
pub use cuda::AprV2ModelCuda;
#[cfg(feature = "cuda")]
use helpers::transpose_matrix;
use helpers::{apply_rope_norm, matmul, rms_norm, simple_attention};
pub use helpers::{detect_format, is_apr_file, simd_dot};
use tokenizer::bpe_encode;
pub use tokenizer::{byte_to_bpe_char, BpeTokenizer, SimpleTokenizer};

/// Magic number: "APR" followed by version byte
/// - Legacy: APR\0 (0x41, 0x50, 0x52, 0x00)
/// - v1: APR1 (0x41, 0x50, 0x52, 0x31)
/// - v2: APR2 (0x41, 0x50, 0x52, 0x32)
pub const MAGIC_PREFIX: [u8; 3] = [0x41, 0x50, 0x52]; // "APR"

/// Legacy magic for compatibility
pub const MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x00];

/// Header size in bytes (64-byte aligned)
pub const HEADER_SIZE: usize = 64;

/// Tensor alignment in bytes
pub const ALIGNMENT: usize = 64;

/// APR v2 feature flags
#[derive(Debug, Clone, Copy, Default)]
pub struct AprFlags(u16);

impl AprFlags {
    /// LZ4 compression enabled
    pub const LZ4_COMPRESSED: u16 = 0x0001;
    /// Zstandard compression enabled
    pub const ZSTD_COMPRESSED: u16 = 0x0002;
    /// Model is encrypted
    pub const ENCRYPTED: u16 = 0x0004;
    /// Model has cryptographic signature
    pub const SIGNED: u16 = 0x0008;
    /// Model is sharded across multiple files
    pub const SHARDED: u16 = 0x0010;
    /// Weights are quantized (int8/int4)
    pub const QUANTIZED: u16 = 0x0020;
    /// Model includes embedded vocabulary
    pub const HAS_VOCAB: u16 = 0x0200;

    /// Create flags from raw bits
    #[must_use]
    pub const fn new(bits: u16) -> Self {
        Self(bits)
    }

    /// Check if model uses compression (LZ4 or Zstd)
    #[must_use]
    pub const fn is_compressed(&self) -> bool {
        self.0 & (Self::LZ4_COMPRESSED | Self::ZSTD_COMPRESSED) != 0
    }

    /// Check if model uses LZ4 compression
    #[must_use]
    pub const fn is_lz4(&self) -> bool {
        self.0 & Self::LZ4_COMPRESSED != 0
    }

    /// Check if model uses ZSTD compression
    #[must_use]
    pub const fn is_zstd(&self) -> bool {
        self.0 & Self::ZSTD_COMPRESSED != 0
    }

    /// Check if model is encrypted
    #[must_use]
    pub const fn is_encrypted(&self) -> bool {
        self.0 & Self::ENCRYPTED != 0
    }

    /// Check if weights are quantized
    #[must_use]
    pub const fn is_quantized(&self) -> bool {
        self.0 & Self::QUANTIZED != 0
    }

    /// Check if model includes embedded vocabulary
    #[must_use]
    pub const fn has_vocab(&self) -> bool {
        self.0 & Self::HAS_VOCAB != 0
    }
}

/// APR v2 file header (64 bytes)
#[derive(Debug, Clone)]
pub struct AprHeader {
    /// Magic number ("APR\0")
    pub magic: [u8; 4],
    /// Format version (major, minor)
    pub version: (u8, u8),
    /// Feature flags
    pub flags: AprFlags,
    /// Number of tensors
    pub tensor_count: u32,
    /// Offset to metadata section
    pub metadata_offset: u64,
    /// Size of metadata section
    pub metadata_size: u32,
    /// Offset to tensor index
    pub tensor_index_offset: u64,
    /// Offset to tensor data
    pub data_offset: u64,
    /// Header checksum (CRC32)
    pub checksum: u32,
}

impl AprHeader {
    /// Parse header from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < HEADER_SIZE {
            return Err(RealizarError::FormatError {
                reason: format!(
                    ".apr header too small: {} bytes (need {})",
                    data.len(),
                    HEADER_SIZE
                ),
            });
        }

        // Check magic - first 3 bytes must be "APR", 4th byte is version
        let magic: [u8; 4] = data[0..4]
            .try_into()
            .map_err(|_| RealizarError::FormatError {
                reason: "Failed to read magic bytes".to_string(),
            })?;

        // Validate magic prefix (APR)
        if magic.get(0..3).expect("magic is 4 bytes") != MAGIC_PREFIX {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "Invalid .apr magic: expected APR {:?}, got {:?}",
                    MAGIC_PREFIX,
                    magic.get(0..3).expect("magic is 4 bytes"),
                ),
            });
        }

        // Validate version byte (0, '1', or '2')
        let version_byte = magic[3];
        if version_byte != 0 && version_byte != b'1' && version_byte != b'2' {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "Invalid .apr version byte: expected 0, '1', or '2', got {}",
                    version_byte
                ),
            });
        }

        // APR v1 (magic "APR1") has different header layout - not supported for inference
        // APR v1 is used by Whisper models but has inline tensor index format
        if version_byte == b'1' {
            return Err(RealizarError::UnsupportedOperation {
                operation: "load_apr_v1".to_string(),
                reason: "APR v1 format not supported for inference. \
                        Use 'apr convert model.apr -o model_v2.apr --format apr2' \
                        to convert to APR v2 format, or use the GGUF version."
                    .to_string(),
            });
        }

        let version = (data[4], data[5]);
        let flags = AprFlags::new(u16::from_le_bytes([data[6], data[7]]));
        let tensor_count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let metadata_offset = u64::from_le_bytes([
            data[12], data[13], data[14], data[15], data[16], data[17], data[18], data[19],
        ]);
        let metadata_size = u32::from_le_bytes([data[20], data[21], data[22], data[23]]);
        let tensor_index_offset = u64::from_le_bytes([
            data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31],
        ]);
        let data_offset = u64::from_le_bytes([
            data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39],
        ]);
        let checksum = u32::from_le_bytes([data[40], data[41], data[42], data[43]]);

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
        })
    }
}

/// Tensor entry in the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    /// Tensor name (e.g., "model.layers.0.attention.wq")
    pub name: String,
    /// Data type (e.g., "F32", "F16", "BF16", "I8")
    pub dtype: String,
    /// Tensor dimensions
    pub shape: Vec<usize>,
    /// Byte offset from data section start
    pub offset: u64,
    /// Size in bytes
    pub size: u64,
}

impl TensorEntry {
    /// Parse tensor entry from binary format (aprender v2 format)
    ///
    /// Binary format:
    /// - name_len (2 bytes LE) + name bytes
    /// - dtype (1 byte)
    /// - ndim (1 byte) + dims (8 bytes LE each, up to 8)
    /// - offset (8 bytes LE)
    /// - size (8 bytes LE)
    pub fn from_binary(data: &[u8]) -> Result<(Self, usize)> {
        if data.len() < 4 {
            return Err(RealizarError::FormatError {
                reason: "Tensor entry too short".to_string(),
            });
        }

        let mut pos = 0;

        // Name
        let name_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;

        if data.len() < pos + name_len + 2 {
            return Err(RealizarError::FormatError {
                reason: "Tensor entry truncated at name".to_string(),
            });
        }

        let name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
        pos += name_len;

        // Dtype (1 byte)
        // GH-191 FIX / GH-321: Use unified GgmlQuantType enum
        // GH-438: APR-native types (AprQ4=128, AprQ8=129) are outside GGML range.
        //   Legacy IDs 8/9 also map to APR-native Q4/Q8 for backwards compat.
        let dtype_byte = data[pos];
        pos += 1;
        let dtype = match dtype_byte {
            // GH-438: APR-native quantization types (new IDs, >=128)
            128 => "q4".to_string(), // AprQ4: per-32-block [f16 scale + 16 nibble bytes]
            129 => "q8".to_string(), // AprQ8: [f32 scale] + [i8 x N]
            // GH-438: Legacy APR-native IDs (collided with GGML Q8_0=8, Q8_1=9)
            8 => "q4".to_string(), // Legacy AprQ4 (was Q4=8 before GH-438)
            9 => "q8".to_string(), // Legacy AprQ8 (was Q8=9 before GH-438)
            // GGML-standard types
            _ => crate::gguf::GgmlQuantType::from_id(u32::from(dtype_byte))
                .map_or_else(
                    || {
                        eprintln!("WARN: Unknown APR dtype byte {dtype_byte}, treating as F32");
                        "F32"
                    },
                    crate::gguf::GgmlQuantType::as_str,
                )
                .to_string(),
        };

        // Shape: ndim (1 byte) + dims
        let ndim = data[pos] as usize;
        pos += 1;

        if data.len() < pos + ndim * 8 + 16 {
            return Err(RealizarError::FormatError {
                reason: "Tensor entry truncated at shape".to_string(),
            });
        }

        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let dim = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]) as usize;
            pos += 8;
            shape.push(dim);
        }

        // Offset and size
        let offset = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]);
        pos += 8;

        let size = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]);
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

    /// Calculate element count from shape
    pub fn element_count(&self) -> usize {
        contract_pre_element_count!();
        let result = self.shape.iter().product();
        contract_post_element_count!(&result);
        result
    }
}

include!("metadata.rs");
include!("tokenizer_loading.rs");
include!("special_tokens.rs");
