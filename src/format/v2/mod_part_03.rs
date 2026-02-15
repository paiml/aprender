
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
