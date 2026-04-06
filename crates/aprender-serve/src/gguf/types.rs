//! GGUF type definitions and constants
//!
//! Core types used throughout the GGUF parser and model loading.
//!
//! This module defines the foundational types for GGUF file parsing:
//! - Magic numbers and version constants
//! - Quantization type constants (Q4_0, Q4_K, Q6_K, etc.)
//! - Buffer types with small buffer optimization
//! - Core structs: GGUFValue, GGUFHeader, TensorInfo, GGUFModel

use std::collections::HashMap;

// ============================================================================
// GGUF Magic and Version Constants
// ============================================================================

/// GGUF magic number: "GGUF" in little-endian
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// GGUF version 2 (same structure as v3, no big-endian)
pub const GGUF_VERSION_V2: u32 = 2;
/// GGUF version 3 (current, adds big-endian support)
pub const GGUF_VERSION_V3: u32 = 3;

// ============================================================================
// Quantization Type Constants
// ============================================================================

/// GGUF quantization type: F32 (unquantized float32)
pub const GGUF_TYPE_F32: u32 = 0;

/// GGUF quantization type: F16 (half precision float16)
pub const GGUF_TYPE_F16: u32 = 1;

/// GGUF quantization type: `Q4_0` (4-bit quantization, block size 32)
pub const GGUF_TYPE_Q4_0: u32 = 2;

/// GGUF quantization type: `Q4_1` (4-bit quantization with min, block size 32)
pub const GGUF_TYPE_Q4_1: u32 = 3;

/// GGUF quantization type: `Q5_0` (5-bit quantization, block size 32)
pub const GGUF_TYPE_Q5_0: u32 = 6;

/// GGUF quantization type: `Q5_1` (5-bit quantization with min, block size 32)
pub const GGUF_TYPE_Q5_1: u32 = 7;

/// GGUF quantization type: `Q8_0` (8-bit quantization, block size 32)
pub const GGUF_TYPE_Q8_0: u32 = 8;

/// GGUF quantization type: `Q2_K` (2-bit K-quantization, super-block size 256)
pub const GGUF_TYPE_Q2_K: u32 = 10;

/// GGUF quantization type: `Q3_K` (3-bit K-quantization, super-block size 256)
pub const GGUF_TYPE_Q3_K: u32 = 11;

/// GGUF quantization type: `Q4_K` (4-bit K-quantization, super-block size 256)
pub const GGUF_TYPE_Q4_K: u32 = 12;

/// GGUF quantization type: `Q5_K` (5-bit K-quantization, super-block size 256)
pub const GGUF_TYPE_Q5_K: u32 = 13;

/// GGUF quantization type: `Q6_K` (6-bit K-quantization, super-block size 256)
pub const GGUF_TYPE_Q6_K: u32 = 14;

/// GGUF quantization type: BF16 (bfloat16, GGML type 30)
pub const GGUF_TYPE_BF16: u32 = 30;

// ============================================================================
// GH-321: Unified GGML Quantization Type Enum
// ============================================================================

/// GGML quantization type — single source of truth for ID<->string mapping.
///
/// CRITICAL (GH-438): These IDs MUST match the GGML standard (llama.cpp ggml.h).
/// aprender's `TensorDType` IDs in the 0-31 range must also match.
/// APR-native quantization types (AprQ4=128, AprQ8=129 in aprender) are
/// outside this enum's scope — they are handled by `TensorEntry::from_binary`
/// before this enum is consulted. See LAYOUT-002.
///
/// Replaces 6+ duplicated match statements across infer/mod.rs, gguf/dtype.rs,
/// apr/mod.rs, apr/dequant.rs, and apr/special_tokens.rs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlQuantType {
    /// 32-bit float (GGML type 0)
    F32 = 0,
    /// 16-bit float (GGML type 1)
    F16 = 1,
    /// 4-bit quantization, block size 32 (GGML type 2)
    Q4_0 = 2,
    /// 4-bit quantization with min, block size 32 (GGML type 3)
    Q4_1 = 3,
    /// 5-bit quantization, block size 32 (GGML type 6)
    Q5_0 = 6,
    /// 5-bit quantization with min, block size 32 (GGML type 7)
    Q5_1 = 7,
    /// 8-bit quantization, block size 32 (GGML type 8)
    Q8_0 = 8,
    /// 8-bit quantization type 1 (GGML type 9)
    Q8_1 = 9,
    /// 2-bit K-quantization, super-block size 256 (GGML type 10)
    Q2K = 10,
    /// 3-bit K-quantization, super-block size 256 (GGML type 11)
    Q3K = 11,
    /// 4-bit K-quantization, super-block size 256 (GGML type 12)
    Q4K = 12,
    /// 5-bit K-quantization, super-block size 256 (GGML type 13)
    Q5K = 13,
    /// 6-bit K-quantization, super-block size 256 (GGML type 14)
    Q6K = 14,
    /// Importance-weighted 2-bit, extra-extra-small (GGML type 16)
    IQ2XXS = 16,
    /// Importance-weighted 2-bit, extra-small (GGML type 17)
    IQ2XS = 17,
    /// BFloat16 (GGML type 30)
    BF16 = 30,
}

impl GgmlQuantType {
    /// Convert from u32 GGML type ID. Returns `None` for unknown types.
    #[must_use]
    pub const fn from_id(id: u32) -> Option<Self> {
        match id {
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
            16 => Some(Self::IQ2XXS),
            17 => Some(Self::IQ2XS),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Convert to u32 GGML type ID.
    #[must_use]
    pub const fn as_id(self) -> u32 {
        self as u32
    }

    /// Convert to u8 for APR binary format.
    #[must_use]
    pub const fn as_byte(self) -> u8 {
        self as u8
    }

    /// Convert to canonical string representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::IQ2XXS => "IQ2_XXS",
            Self::IQ2XS => "IQ2_XS",
            Self::BF16 => "BF16",
        }
    }

    /// Parse from string (case-insensitive). Returns `None` for unknown strings.
    #[must_use]
    pub fn from_str_lossy(s: &str) -> Option<Self> {
        match s {
            "F32" | "f32" => Some(Self::F32),
            "F16" | "f16" => Some(Self::F16),
            "BF16" | "bf16" => Some(Self::BF16),
            "Q4_0" | "q4_0" => Some(Self::Q4_0),
            "Q4_1" | "q4_1" => Some(Self::Q4_1),
            "Q5_0" | "q5_0" => Some(Self::Q5_0),
            "Q5_1" | "q5_1" => Some(Self::Q5_1),
            "Q8_0" | "q8_0" => Some(Self::Q8_0),
            "Q8_1" | "q8_1" => Some(Self::Q8_1),
            "Q2_K" | "q2_k" => Some(Self::Q2K),
            "Q3_K" | "q3_k" => Some(Self::Q3K),
            "Q4_K" | "q4_k" => Some(Self::Q4K),
            "Q5_K" | "q5_k" => Some(Self::Q5K),
            "Q6_K" | "q6_k" => Some(Self::Q6K),
            "IQ2_XXS" | "iq2_xxs" => Some(Self::IQ2XXS),
            "IQ2_XS" | "iq2_xs" => Some(Self::IQ2XS),
            _ => None,
        }
    }
}

impl std::fmt::Display for GgmlQuantType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ============================================================================
// IMP-117: Small Buffer Optimization Constants (per spec Section 4.1-4.2)
// ============================================================================

/// Small buffer inline capacity for token IDs (IMP-117)
/// Most prompts are < 32 tokens, avoiding heap allocation
pub const TOKEN_BUFFER_INLINE_CAP: usize = 32;

/// Small buffer inline capacity for attention scores (IMP-117)
/// Stack-allocated for short sequences (per-head, small context)
pub const ATTENTION_BUFFER_INLINE_CAP: usize = 64;

/// Small buffer inline capacity for hidden states (IMP-117)
/// Inline storage for small models (hidden_dim <= 128)
pub const HIDDEN_BUFFER_INLINE_CAP: usize = 128;

/// Buffer watermark: Low mark for inline/stack allocation
pub const BUFFER_LW_SIZE: usize = 1024;

/// Buffer watermark: High mark for pooled allocations
pub const BUFFER_HW_SIZE: usize = 8 * 1024;

/// Buffer watermark: Maximum before chunking
pub const BUFFER_MAX_SIZE: usize = 32 * 1024;

// ============================================================================
// Buffer Type Aliases with Small Buffer Optimization
// ============================================================================

/// Token buffer with inline storage (IMP-117)
/// Uses SmallVec for stack allocation when size <= TOKEN_BUFFER_INLINE_CAP
pub type TokenBuffer = smallvec::SmallVec<[u32; TOKEN_BUFFER_INLINE_CAP]>;

/// Attention score buffer with inline storage (IMP-117)
/// Uses SmallVec for stack allocation when size <= ATTENTION_BUFFER_INLINE_CAP
pub type AttentionBuffer = smallvec::SmallVec<[f32; ATTENTION_BUFFER_INLINE_CAP]>;

/// Hidden state buffer with inline storage (IMP-117)
/// Uses SmallVec for stack allocation when size <= HIDDEN_BUFFER_INLINE_CAP
pub type HiddenBuffer = smallvec::SmallVec<[f32; HIDDEN_BUFFER_INLINE_CAP]>;

// ============================================================================
// Core GGUF Types
// ============================================================================

/// GGUF alignment requirement (32 bytes)
pub const GGUF_ALIGNMENT: usize = 32;

/// GGUF metadata value types
#[derive(Debug, Clone, PartialEq)]
pub enum GGUFValue {
    /// Unsigned 8-bit integer
    UInt8(u8),
    /// Signed 8-bit integer
    Int8(i8),
    /// Unsigned 16-bit integer
    UInt16(u16),
    /// Signed 16-bit integer
    Int16(i16),
    /// Unsigned 32-bit integer
    UInt32(u32),
    /// Signed 32-bit integer
    Int32(i32),
    /// 32-bit floating point
    Float32(f32),
    /// Boolean
    Bool(bool),
    /// UTF-8 string
    String(String),
    /// Array of values
    Array(Vec<GGUFValue>),
    /// Unsigned 64-bit integer
    UInt64(u64),
    /// Signed 64-bit integer
    Int64(i64),
    /// 64-bit floating point
    Float64(f64),
}

/// GGUF file header
#[derive(Debug, Clone, PartialEq)]
pub struct GGUFHeader {
    /// Magic number (must be `GGUF_MAGIC`)
    pub magic: u32,
    /// Format version
    pub version: u32,
    /// Number of tensors in the file
    pub tensor_count: u64,
    /// Number of metadata key-value pairs
    pub metadata_count: u64,
}

/// Tensor information
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    /// Tensor name
    pub name: String,
    /// Number of dimensions
    pub n_dims: u32,
    /// Dimensions (shape)
    pub dims: Vec<u64>,
    /// Quantization type
    pub qtype: u32,
    /// Offset in the file where tensor data starts
    pub offset: u64,
}

/// GGUF model container
#[derive(Debug, Clone)]
pub struct GGUFModel {
    /// File header
    pub header: GGUFHeader,
    /// Metadata key-value pairs
    pub metadata: HashMap<String, GGUFValue>,
    /// Tensor information
    pub tensors: Vec<TensorInfo>,
    /// Offset where tensor data starts (after header/metadata/tensor_info + alignment)
    pub tensor_data_start: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_constant() {
        assert_eq!(GGUF_MAGIC, 0x4655_4747);
    }

    #[test]
    fn test_quantization_constants() {
        assert_eq!(GGUF_TYPE_F32, 0);
        assert_eq!(GGUF_TYPE_F16, 1);
        assert_eq!(GGUF_TYPE_Q4_0, 2);
        assert_eq!(GGUF_TYPE_Q8_0, 8);
        assert_eq!(GGUF_TYPE_Q4_K, 12);
        assert_eq!(GGUF_TYPE_Q6_K, 14);
    }

    #[test]
    fn test_buffer_constants() {
        assert_eq!(TOKEN_BUFFER_INLINE_CAP, 32);
        assert_eq!(ATTENTION_BUFFER_INLINE_CAP, 64);
        assert_eq!(HIDDEN_BUFFER_INLINE_CAP, 128);
    }

    #[test]
    fn test_buffer_watermarks() {
        assert_eq!(BUFFER_LW_SIZE, 1024);
        assert_eq!(BUFFER_HW_SIZE, 8 * 1024);
        assert_eq!(BUFFER_MAX_SIZE, 32 * 1024);
    }

    #[test]
    fn test_version_constant() {
        assert_eq!(GGUF_VERSION_V3, 3);
    }

    #[test]
    fn test_gguf_value_variants() {
        let uint8 = GGUFValue::UInt8(255);
        let string = GGUFValue::String("test".to_string());
        let array = GGUFValue::Array(vec![GGUFValue::UInt32(1), GGUFValue::UInt32(2)]);

        assert_eq!(uint8, GGUFValue::UInt8(255));
        assert_eq!(string, GGUFValue::String("test".to_string()));
        assert!(matches!(array, GGUFValue::Array(_)));
    }

    #[test]
    fn test_gguf_header() {
        let header = GGUFHeader {
            magic: GGUF_MAGIC,
            version: GGUF_VERSION_V3,
            tensor_count: 100,
            metadata_count: 50,
        };

        assert_eq!(header.magic, 0x4655_4747);
        assert_eq!(header.version, 3);
        assert_eq!(header.tensor_count, 100);
        assert_eq!(header.metadata_count, 50);
    }

    #[test]
    fn test_tensor_info() {
        let info = TensorInfo {
            name: "model.layers.0.attn.wq".to_string(),
            n_dims: 2,
            dims: vec![4096, 4096],
            qtype: GGUF_TYPE_Q4_K,
            offset: 1024,
        };

        assert_eq!(info.name, "model.layers.0.attn.wq");
        assert_eq!(info.n_dims, 2);
        assert_eq!(info.dims, vec![4096, 4096]);
        assert_eq!(info.qtype, GGUF_TYPE_Q4_K);
        assert_eq!(info.offset, 1024);
    }

    #[test]
    fn test_gguf_model() {
        let model = GGUFModel {
            header: GGUFHeader {
                magic: GGUF_MAGIC,
                version: GGUF_VERSION_V3,
                tensor_count: 1,
                metadata_count: 0,
            },
            metadata: HashMap::new(),
            tensors: vec![],
            tensor_data_start: 128,
        };

        assert_eq!(model.header.magic, GGUF_MAGIC);
        assert!(model.tensors.is_empty());
        assert_eq!(model.tensor_data_start, 128);
    }

    #[test]
    fn test_alignment_constant() {
        assert_eq!(GGUF_ALIGNMENT, 32);
        // Verify it's a power of 2
        assert_eq!(GGUF_ALIGNMENT & (GGUF_ALIGNMENT - 1), 0);
    }
}
