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

/// APR-native quantization type: Q4 (4-bit, block size 32, f16 scale per block).
///
/// GH-478: Distinct from `GGUF_TYPE_Q4_0` because the byte layout differs
/// (APR Q4 has the `i % 2 == 0 → low nibble, i % 2 == 1 → high nibble` split,
/// not the GGML "first 16 in low, last 16 in high" arrangement). Stored in
/// `OwnedQuantizedTensor.qtype` so `fused_matmul` can dispatch to the per-tensor
/// scratch dequant path instead of expanding all weights to F32 at load.
///
/// Matches `aprender::format::TensorDType::Q4 = 128` (file-format value).
pub const APR_TYPE_Q4: u32 = 128;

/// APR-native quantization type: Q8 (8-bit, block size 32, f16 scale per block).
///
/// GH-478: Same role as `APR_TYPE_Q4` for the 8-bit native format.
/// Matches `aprender::format::TensorDType::Q8 = 129` (file-format value).
pub const APR_TYPE_Q8: u32 = 129;
// PMAT-3430 M1 Phase 4: the enum that used to be declared here is now the
// workspace's ONE ggml tensor-type enum, re-exported from the leaf crate.
//
// serve's copy listed 16 ids and was the only one of the three with string
// conversions and a Display impl. Those move into the leaf with unchanged names
// and unchanged behaviour: `as_str` returns the same 16 GGUF spellings, and
// `from_str_lossy` still accepts exactly the exact spelling and its all-lower
// form (so ggml's own `q4_K` is still refused, as it always was).
//
// The NAME survives as an alias because roughly thirty call sites and several
// public signatures use it, and renaming them is not what M1 is for.
pub use trueno_quant::GgmlType;
/// serve's historical name for the ggml tensor-type enum.
pub use trueno_quant::GgmlType as GgmlQuantType;

/// The ids serve ACCEPTS at its parse boundaries.
///
/// The leaf knows all 35 live ggml types; serve knew 16 and must keep knowing
/// exactly 16 (#3430 Q1-c). Every boundary in this crate that turns an id or a
/// name into a type goes through [`admitted_from_id`] or [`admitted_from_name`]
/// rather than through the leaf directly, so the set is stated once and cannot
/// widen by accident when the enum grows.
///
/// A type in this list is not a promise that serve can DECODE it — `apr/dequant`
/// narrows further, to the quantized subset it has a dequantiser for.
pub const ADMITTED: [GgmlQuantType; 16] = [
    GgmlQuantType::F32,
    GgmlQuantType::F16,
    GgmlQuantType::Q4_0,
    GgmlQuantType::Q4_1,
    GgmlQuantType::Q5_0,
    GgmlQuantType::Q5_1,
    GgmlQuantType::Q8_0,
    GgmlQuantType::Q8_1,
    GgmlQuantType::Q2K,
    GgmlQuantType::Q3K,
    GgmlQuantType::Q4K,
    GgmlQuantType::Q5K,
    GgmlQuantType::Q6K,
    GgmlQuantType::IQ2XXS,
    GgmlQuantType::IQ2XS,
    GgmlQuantType::BF16,
];

/// Is this type one serve admits?
#[must_use]
pub const fn is_admitted(t: GgmlQuantType) -> bool {
    let mut i = 0;
    while i < ADMITTED.len() {
        if ADMITTED[i] as u32 == t as u32 {
            return true;
        }
        i += 1;
    }
    false
}

/// An id becomes a type only if serve admits it. The replacement for a bare
/// `GgmlQuantType::from_id` at every parse boundary in this crate.
#[must_use]
pub const fn admitted_from_id(id: u32) -> Option<GgmlQuantType> {
    match GgmlQuantType::from_id(id) {
        Some(t) if is_admitted(t) => Some(t),
        _ => None,
    }
}

/// A name becomes a type only if serve admits it. The replacement for a bare
/// `GgmlQuantType::from_str_lossy`.
#[must_use]
pub fn admitted_from_name(name: &str) -> Option<GgmlQuantType> {
    GgmlQuantType::from_str_lossy(name).filter(|t| is_admitted(*t))
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
