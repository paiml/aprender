//! Tensor-index entry impl, `TensorDType`, and 64-byte alignment utilities
//! (issue #2231). Formerly `include!`d into `v2/mod.rs`; now a real module.

use super::{TensorIndexEntry, V2FormatError, ALIGNMENT, MAX_TENSOR_NAME_LEN};

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

/// Tensor data type for APR v2 format.
///
/// # GGML Standard Compliance (GH-438)
///
/// CRITICAL: IDs in the 0–31 range MUST match the GGML standard (llama.cpp ggml.h).
/// realizar's `GgmlQuantType::from_id()` decodes these bytes directly.
///
/// GGML standard reference (authoritative: ggml.h `enum ggml_type`):
///   F32=0, F16=1, Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_0=8, Q8_1=9,
///   Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15,
///   IQ2_XXS=16, IQ2_XS=17, ..., BF16=30
///
/// APR-native quantization types (AprQ4, AprQ8) use IDs >= 128 to avoid
/// collision with the GGML ID space. These have different block formats
/// than any GGML type and must NOT share IDs with GGML types.
///
/// Legacy note: APR files written before GH-438 used IDs 8 (Q4) and 9 (Q8),
/// which collide with GGML Q8_0=8 and Q8_1=9. `from_u8()` accepts both
/// old and new IDs for backwards compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TensorDType {
    /// 32-bit float (GGML type 0)
    F32 = 0,
    /// 16-bit float (GGML type 1)
    F16 = 1,
    /// Brain float 16 (GGML type 30)
    BF16 = 30,
    /// 64-bit float (APR extension, not in GGML)
    F64 = 3,
    /// 32-bit signed integer (APR extension, not in GGML)
    I32 = 4,
    /// 64-bit signed integer (APR extension, not in GGML)
    I64 = 5,
    /// 8-bit signed integer (APR extension, not in GGML)
    I8 = 6,
    /// 8-bit unsigned integer (APR extension, not in GGML)
    U8 = 7,
    /// APR-native 4-bit symmetric block quantization (NOT GGML Q4_0/Q4_K).
    /// Format: per-32-block [scale: f16 (2B)] + [16 packed nibble bytes]
    /// ID 128: outside GGML range to prevent collision.
    /// Legacy: was ID 8 (collided with GGML Q8_0). See GH-438.
    AprQ4 = 128,
    /// APR-native 8-bit single-scale quantization (NOT GGML Q8_0/Q8_1).
    /// Format: [scale: f32 (4B)] + [i8 x N] (single whole-tensor scale)
    /// ID 129: outside GGML range to prevent collision.
    /// Legacy: was ID 9 (collided with GGML Q8_1). See GH-438.
    AprQ8 = 129,
    /// GGUF Q4_K format (GGML type 12, raw super-blocks, ~4.5 bits/weight)
    /// Format: 256-element blocks with super-block scales
    Q4K = 12,
    /// GGUF Q6_K format (GGML type 14, raw super-blocks, ~6.5 bits/weight)
    Q6K = 14,
}

impl std::fmt::Display for TensorDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F64 => "F64",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::I8 => "I8",
            Self::U8 => "U8",
            Self::AprQ4 => "APR_Q4",
            Self::AprQ8 => "APR_Q8",
            Self::Q4K => "Q4_K",
            Self::Q6K => "Q6_K",
        };
        f.write_str(name)
    }
}

// ============================================================================
// Compile-time assertions: GGML-aligned IDs must match the standard (GH-438)
// ============================================================================
const _: () = assert!(TensorDType::F32 as u8 == 0, "F32 must be GGML type 0");
const _: () = assert!(TensorDType::F16 as u8 == 1, "F16 must be GGML type 1");
const _: () = assert!(TensorDType::BF16 as u8 == 30, "BF16 must be GGML type 30");
const _: () = assert!(
    TensorDType::Q4K as u8 == 12,
    "Q4K must be GGML type 12 (Q4_K)"
);
const _: () = assert!(
    TensorDType::Q6K as u8 == 14,
    "Q6K must be GGML type 14 (Q6_K)"
);
// APR-native types must be outside GGML range (>=128)
const _: () = assert!(
    TensorDType::AprQ4 as u8 >= 128,
    "AprQ4 must be outside GGML range"
);
const _: () = assert!(
    TensorDType::AprQ8 as u8 >= 128,
    "AprQ8 must be outside GGML range"
);

impl TensorDType {
    /// Convert from u8.
    ///
    /// Accepts both current IDs (128=AprQ4, 129=AprQ8) and legacy IDs
    /// (8=AprQ4, 9=AprQ8) for backwards compatibility with pre-GH-438 APR files.
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            30 => Some(Self::BF16),
            3 => Some(Self::F64),
            4 => Some(Self::I32),
            5 => Some(Self::I64),
            6 => Some(Self::I8),
            7 => Some(Self::U8),
            // GH-438: Legacy IDs 8/9 (collided with GGML Q8_0/Q8_1)
            8 | 128 => Some(Self::AprQ4),
            9 | 129 => Some(Self::AprQ8),
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
            Self::I8 | Self::U8 | Self::AprQ8 => 1,
            Self::AprQ4 | Self::Q4K | Self::Q6K => 0, // Packed/block formats, need special handling
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
            Self::AprQ4 => "q4",
            Self::AprQ8 => "q8",
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
    value.is_multiple_of(ALIGNMENT)
}

// The `AprV2Writer` / `AprV2StreamingWriter` / `AprV2Reader` / `AprV2ReaderRef`
// struct declarations now live alongside their `impl` blocks (in `writer.rs` and
// `streaming_writer.rs`) so private-field access stays module-local after the
// include!()→mod split (issue #2231).
