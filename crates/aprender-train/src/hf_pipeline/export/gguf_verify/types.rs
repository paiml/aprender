//! GGUF verification result types

/// Summary of a verified GGUF file
#[derive(Debug, Clone)]
pub struct GgufSummary {
    /// GGUF format version
    pub version: u32,
    /// Number of tensors
    pub tensor_count: u64,
    /// Number of metadata key-value pairs
    pub metadata_count: u64,
    /// Total file size in bytes
    pub file_size: usize,
    /// Tensor names and their data types (as GGUF type IDs)
    pub tensors: Vec<GgufTensorInfo>,
}

/// Info about a single tensor in the GGUF file
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name
    pub name: String,
    /// Shape dimensions
    pub shape: Vec<u64>,
    /// GGUF data type ID
    pub dtype: u32,
    /// Byte offset in file
    pub offset: u64,
}
