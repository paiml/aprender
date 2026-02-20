
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
mod tests;
