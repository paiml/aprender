//! APR Format Module (v2 `APR\0`) — sovereign leaf (issue #2231)
//!
//! Implements the APR v2 container format with:
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
//! use apr_format::v2::{AprV2Header, AprV2Flags, MAGIC_V2, ALIGNMENT};
//!
//! let header = AprV2Header::new();
//! assert_eq!(header.magic, MAGIC_V2);
//! assert!(header.is_valid());
//! ```
//!
//! # Sovereignty (issue #2231)
//!
//! This module contains ONLY the container I/O — pure bytes, shapes, and
//! dtypes. It carries **no** ML/GPU/tokenizer dependency:
//!   - CRC32 routes through the single [`crate::crc32::crc32`].
//!   - f16 conversion routes through [`crate::f16`] (the IEEE-correct `half`
//!     crate), NOT `trueno::f32_to_f16`. See the f16 note in `crate::f16`.
//!   - The dequantizing `get_tensor_as_f32` accessor (which needs the GGUF
//!     Q4_K/Q6_K dequant + f32 physics) is **severed** from the leaf reader and
//!     re-attached in `aprender-core` as an extension trait (`AprV2DequantExt`).
//!     The leaf exposes the raw bytes via [`AprV2Reader::get_tensor_data`] and
//!     the typed-but-trivial [`AprV2Reader::get_f32_tensor`] (F32 dtype only).

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

// --- Split modules (was include!(); now real `mod`s, issue #2231 Stage 2) ----
// Each formerly-`include!`d file is a real module that re-derives its own `use`
// block and reaches the parent-scope structs/consts via `super::`. The public
// items are re-exported into the `v2` namespace below so the historical flat
// paths (`aprender::format::v2::AprV2Reader`, …) keep resolving unchanged.
mod header_impl;
mod reader_impl;
mod streaming_writer;
mod tensor_index_impl;
mod v2format_error;
mod writer;

pub use header_impl::{
    AprV2Metadata, ChatSpecialTokens, QuantizationMetadata, ShardingMetadata, TensorIndexEntry,
};
pub use reader_impl::{required_file_len, AprV2Reader, AprV2ReaderRef, ShardInfo, ShardManifest};
pub use streaming_writer::AprV2StreamingWriter;
pub use tensor_index_impl::{align_64, align_up, is_aligned_64, padding_to_align, TensorDType};
pub use v2format_error::V2FormatError;
pub use writer::AprV2Writer;

// Provenance stamping — SHIP-009 full-discharge enabler (task #141).
// Lives as a real submodule (not `include!`) so its inline tests nest
// cleanly under `v2::stamp::tests`.
pub mod stamp;
pub use stamp::{stamp_provenance_bytes, ProvenancePatch};

#[cfg(test)]
mod tests;

// Issue #2612: the data-extent invariant falsifiers.
#[cfg(test)]
mod tests_truncation_2612;
