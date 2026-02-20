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
///
/// ONE PATH: Delegates to `trueno::f32_to_f16` (UCBD §4).
fn f32_to_f16(value: f32) -> u16 {
    trueno::f32_to_f16(value)
}

/// Convert f16 to f32
///
/// ONE PATH: Delegates to `trueno::f16_to_f32` (UCBD §4).
fn f16_to_f32(bits: u16) -> f32 {
    trueno::f16_to_f32(bits)
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
        let scale =
            if scale_raw.is_nan() || scale_raw.is_infinite() || scale_raw.abs() < F16_MIN_NORMAL {
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

    /// LAYOUT-002: Tensor layout is row-major (REQUIRED for valid APR files)
    /// All APR files created after LAYOUT-002 must have this flag set.
    /// Pre-LAYOUT-002 files without this flag are assumed row-major.
    pub const LAYOUT_ROW_MAJOR: u16 = 0b0000_0100_0000_0000;

    /// LAYOUT-002: Tensor layout is column-major (FORBIDDEN - Jidoka guard)
    /// If this flag is set, the APR file is "dirty" and must be rejected.
    /// This flag exists to catch improperly converted GGUF files.
    pub const LAYOUT_COLUMN_MAJOR: u16 = 0b0000_1000_0000_0000;

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

    /// LAYOUT-002: Check if row-major layout flag is set
    #[must_use]
    pub const fn is_row_major(self) -> bool {
        self.contains(Self::LAYOUT_ROW_MAJOR)
    }

    /// LAYOUT-002: Check if column-major layout flag is set (should be rejected)
    #[must_use]
    pub const fn is_column_major(self) -> bool {
        self.contains(Self::LAYOUT_COLUMN_MAJOR)
    }

    /// LAYOUT-002: Validate layout is safe for inference
    /// Returns true if the file is row-major or pre-LAYOUT-002 (assumed row-major)
    /// Returns false if explicitly marked as column-major (dirty APR file)
    #[must_use]
    pub const fn is_layout_valid(self) -> bool {
        // Reject if explicitly marked as column-major
        !self.is_column_major()
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

include!("mod_part_02.rs");
include!("mod_part_03.rs");
include!("writer.rs");
include!("mod_part_05.rs");
include!("v2format_error.rs");
