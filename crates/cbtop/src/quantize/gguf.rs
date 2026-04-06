//! GGUF file format parsing types and loader.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

/// GGUF file header (simplified parsing).
///
/// GGUF format specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
#[derive(Debug, Clone)]
pub struct GgufHeader {
    /// Magic number ("GGUF")
    pub magic: [u8; 4],
    /// Format version
    pub version: u32,
    /// Number of tensors
    pub tensor_count: u64,
    /// Number of metadata key-value pairs
    pub metadata_kv_count: u64,
}

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

/// GGUF tensor info.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name
    pub name: String,
    /// Number of dimensions
    pub n_dims: u32,
    /// Dimensions
    pub dims: Vec<u64>,
    /// Data type (GGML type)
    pub dtype: u32,
    /// Offset in data section
    pub offset: u64,
}

/// Result type for GGUF operations.
pub type GgufResult<T> = Result<T, GgufError>;

/// GGUF parsing errors.
#[derive(Debug, Clone)]
pub enum GgufError {
    /// Invalid magic number
    InvalidMagic([u8; 4]),
    /// Unsupported version
    UnsupportedVersion(u32),
    /// IO error
    Io(String),
    /// Invalid data
    InvalidData(String),
    /// Tensor not found
    TensorNotFound(String),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::InvalidMagic(magic) => {
                write!(f, "Invalid GGUF magic: {:?}", magic)
            }
            GgufError::UnsupportedVersion(v) => {
                write!(f, "Unsupported GGUF version: {}", v)
            }
            GgufError::Io(msg) => write!(f, "IO error: {}", msg),
            GgufError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            GgufError::TensorNotFound(name) => write!(f, "Tensor not found: {}", name),
        }
    }
}

impl std::error::Error for GgufError {}

/// GGUF file loader (basic implementation).
#[derive(Debug)]
pub struct GgufLoader {
    /// File path
    path: String,
    /// Header info
    header: Option<GgufHeader>,
    /// Tensor metadata
    tensors: Vec<GgufTensorInfo>,
    /// Model metadata
    metadata: HashMap<String, GgufValue>,
}

impl GgufLoader {
    /// Create a new GGUF loader for a file path.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_string_lossy().to_string(),
            header: None,
            tensors: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Check if path exists and has GGUF extension.
    pub fn validate_path(&self) -> GgufResult<()> {
        let path = Path::new(&self.path);
        if !path.exists() {
            return Err(GgufError::Io(format!("File not found: {}", self.path)));
        }
        if path.extension().map_or(true, |ext| ext != "gguf") {
            return Err(GgufError::InvalidData("File does not have .gguf extension".to_string()));
        }
        Ok(())
    }

    /// Parse GGUF header from bytes.
    pub fn parse_header(&mut self, data: &[u8]) -> GgufResult<()> {
        if data.len() < 24 {
            return Err(GgufError::InvalidData("File too small for header".to_string()));
        }

        // SAFETY: length checked above (data.len() >= 24), so all slices are in bounds
        let magic: [u8; 4] = data[0..4].try_into().expect("invariant: slice is 4 bytes");
        if &magic != b"GGUF" {
            return Err(GgufError::InvalidMagic(magic));
        }

        let version =
            u32::from_le_bytes(data[4..8].try_into().expect("invariant: slice is 4 bytes"));
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count =
            u64::from_le_bytes(data[8..16].try_into().expect("invariant: slice is 8 bytes"));
        let metadata_kv_count =
            u64::from_le_bytes(data[16..24].try_into().expect("invariant: slice is 8 bytes"));

        self.header = Some(GgufHeader { magic, version, tensor_count, metadata_kv_count });

        Ok(())
    }

    /// Get parsed header.
    pub fn header(&self) -> Option<&GgufHeader> {
        self.header.as_ref()
    }

    /// Get tensor count.
    pub fn tensor_count(&self) -> u64 {
        self.header.as_ref().map_or(0, |h| h.tensor_count)
    }

    /// Get file path.
    pub fn path(&self) -> &str {
        &self.path
    }
}
