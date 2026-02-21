
impl AprV2Header {
    /// Create new v2 header with defaults
    #[must_use]
    pub fn new() -> Self {
        Self {
            magic: MAGIC_V2,
            version: VERSION_V2,
            flags: AprV2Flags::new(),
            tensor_count: 0,
            metadata_offset: HEADER_SIZE_V2 as u64,
            metadata_size: 0,
            tensor_index_offset: 0,
            data_offset: 0,
            checksum: 0,
            reserved: [0u8; 20],
        }
    }

    /// Check if header has valid magic number
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.magic == MAGIC_V2
    }

    /// Serialize header to bytes
    #[must_use]
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE_V2] {
        let mut buf = [0u8; HEADER_SIZE_V2];

        buf[0..4].copy_from_slice(&self.magic);
        buf[4] = self.version.0;
        buf[5] = self.version.1;
        buf[6..8].copy_from_slice(&self.flags.bits().to_le_bytes());
        buf[8..12].copy_from_slice(&self.tensor_count.to_le_bytes());
        buf[12..20].copy_from_slice(&self.metadata_offset.to_le_bytes());
        buf[20..24].copy_from_slice(&self.metadata_size.to_le_bytes());
        buf[24..32].copy_from_slice(&self.tensor_index_offset.to_le_bytes());
        buf[32..40].copy_from_slice(&self.data_offset.to_le_bytes());
        buf[40..44].copy_from_slice(&self.checksum.to_le_bytes());
        buf[44..64].copy_from_slice(&self.reserved);

        buf
    }

    /// Deserialize header from bytes
    ///
    /// # Errors
    /// Returns error if buffer is too small or magic is invalid.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, V2FormatError> {
        if buf.len() < HEADER_SIZE_V2 {
            return Err(V2FormatError::InvalidHeader("buffer too small".to_string()));
        }

        let magic: [u8; 4] = buf[0..4]
            .try_into()
            .map_err(|_| V2FormatError::InvalidHeader("failed to read magic".to_string()))?;

        // Check for v2 magic only
        if magic != MAGIC_V2 {
            return Err(V2FormatError::InvalidMagic(magic));
        }

        let version = (buf[4], buf[5]);
        let flags = AprV2Flags::from_bits(u16::from_le_bytes([buf[6], buf[7]]));
        let tensor_count = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let metadata_offset = u64::from_le_bytes(buf[12..20].try_into().unwrap_or([0; 8]));
        let metadata_size = u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]);
        let tensor_index_offset = u64::from_le_bytes(buf[24..32].try_into().unwrap_or([0; 8]));
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap_or([0; 8]));
        let checksum = u32::from_le_bytes([buf[40], buf[41], buf[42], buf[43]]);

        let mut reserved = [0u8; 20];
        reserved.copy_from_slice(buf.get(44..64).unwrap_or(&[0u8; 20]));

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
            reserved,
        })
    }

    /// Compute header checksum (CRC32 of header bytes excluding checksum field)
    #[must_use]
    pub fn compute_checksum(&self) -> u32 {
        let bytes = self.to_bytes();
        // Exclude checksum field (bytes 40-43) from calculation
        // Concatenate the two regions and compute CRC32
        let mut data = Vec::with_capacity(60);
        data.extend_from_slice(bytes.get(0..40).unwrap_or(&[]));
        data.extend_from_slice(bytes.get(44..64).unwrap_or(&[]));
        crc32(&data)
    }

    /// Update checksum field
    pub fn update_checksum(&mut self) {
        self.checksum = self.compute_checksum();
    }

    /// Verify header checksum
    #[must_use]
    pub fn verify_checksum(&self) -> bool {
        self.checksum == self.compute_checksum()
    }
}

// ============================================================================
// Metadata
// ============================================================================

/// APR v2 JSON metadata section
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AprV2Metadata {
    /// Model type identifier
    #[serde(default)]
    pub model_type: String,

    /// Model name
    #[serde(default)]
    pub name: Option<String>,

    /// Model description
    #[serde(default)]
    pub description: Option<String>,

    /// Model author/organization
    #[serde(default)]
    pub author: Option<String>,

    /// Model license
    #[serde(default)]
    pub license: Option<String>,

    /// Model version string
    #[serde(default)]
    pub version: Option<String>,

    /// Source/provenance URI (DD6: Model provenance tracking)
    /// Examples: "<hf://openai/whisper-tiny>", "<local://path/to/model.safetensors>"
    #[serde(default)]
    pub source: Option<String>,

    /// Original format before conversion
    /// Examples: "safetensors", "gguf", "pytorch"
    #[serde(default)]
    pub original_format: Option<String>,

    /// Creation timestamp (ISO 8601)
    #[serde(default)]
    pub created_at: Option<String>,

    /// Total model size in bytes
    #[serde(default)]
    pub total_size: u64,

    /// Parameter count
    #[serde(default)]
    pub param_count: u64,

    /// Quantization info
    #[serde(default)]
    pub quantization: Option<QuantizationMetadata>,

    /// Shard info (for multi-file models)
    #[serde(default)]
    pub sharding: Option<ShardingMetadata>,

    /// Chat template (Jinja2 format, from tokenizer_config.json)
    /// Per spec: chat-template-improvement-spec.md CTA-01
    #[serde(default)]
    pub chat_template: Option<String>,

    /// Detected chat template format
    /// Per spec: chat-template-improvement-spec.md CTA-03
    /// Values: "chatml", "llama2", "mistral", "phi", "alpaca", "custom", "raw"
    #[serde(default)]
    pub chat_format: Option<String>,

    /// Special tokens for chat templates
    /// Per spec: chat-template-improvement-spec.md CTA-04
    #[serde(default)]
    pub special_tokens: Option<ChatSpecialTokens>,

    // ========================================================================
    // Transformer Config (CRITICAL for inference - realizar::apr::AprMetadata)
    // ========================================================================
    /// Model architecture family (e.g., "llama", "qwen2", "phi")
    #[serde(default)]
    pub architecture: Option<String>,

    /// Hidden dimension size
    #[serde(default)]
    pub hidden_size: Option<usize>,

    /// Number of transformer layers
    #[serde(default)]
    pub num_layers: Option<usize>,

    /// Number of attention heads
    #[serde(default)]
    pub num_heads: Option<usize>,

    /// Number of key-value heads (for GQA, defaults to num_heads)
    #[serde(default)]
    pub num_kv_heads: Option<usize>,

    /// Vocabulary size
    #[serde(default)]
    pub vocab_size: Option<usize>,

    /// FFN intermediate dimension
    #[serde(default)]
    pub intermediate_size: Option<usize>,

    /// Maximum context/sequence length
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    /// RoPE theta for position encoding
    #[serde(default)]
    pub rope_theta: Option<f32>,

    /// RoPE type: 0=NORM (adjacent pairs), 2=NEOX (split halves)
    /// CORRECTNESS-011: Qwen2.5 models require rope_type=2 (NEOX style)
    #[serde(default)]
    pub rope_type: Option<u32>,

    /// Layer norm epsilon
    #[serde(default)]
    pub rms_norm_eps: Option<f32>,

    /// Custom key-value pairs
    #[serde(default, flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

/// Special tokens for chat templates (CTA-04)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatSpecialTokens {
    /// Beginning of sequence token
    #[serde(default)]
    pub bos_token: Option<String>,

    /// End of sequence token
    #[serde(default)]
    pub eos_token: Option<String>,

    /// Unknown token
    #[serde(default)]
    pub unk_token: Option<String>,

    /// Padding token
    #[serde(default)]
    pub pad_token: Option<String>,

    /// ChatML start token (<|im_start|>)
    #[serde(default)]
    pub im_start_token: Option<String>,

    /// ChatML end token (<|im_end|>)
    #[serde(default)]
    pub im_end_token: Option<String>,
}

impl AprV2Metadata {
    /// Create new empty metadata
    #[must_use]
    pub fn new(model_type: impl Into<String>) -> Self {
        Self {
            model_type: model_type.into(),
            ..Default::default()
        }
    }

    /// Serialize to JSON bytes
    ///
    /// # Errors
    /// Returns error if serialization fails.
    pub fn to_json(&self) -> Result<Vec<u8>, V2FormatError> {
        serde_json::to_vec(self).map_err(|e| V2FormatError::MetadataError(e.to_string()))
    }

    /// Serialize to pretty JSON string
    ///
    /// # Errors
    /// Returns error if serialization fails.
    pub fn to_json_pretty(&self) -> Result<String, V2FormatError> {
        serde_json::to_string_pretty(self).map_err(|e| V2FormatError::MetadataError(e.to_string()))
    }

    /// Deserialize from JSON bytes
    ///
    /// # Errors
    /// Returns error if deserialization fails.
    pub fn from_json(data: &[u8]) -> Result<Self, V2FormatError> {
        serde_json::from_slice(data).map_err(|e| V2FormatError::MetadataError(e.to_string()))
    }
}

/// Quantization metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuantizationMetadata {
    /// Quantization type (e.g., "int8", "int4", "fp16")
    pub quant_type: String,
    /// Bits per weight
    pub bits: u8,
    /// Block size for block quantization
    pub block_size: Option<usize>,
    /// Whether symmetric quantization
    pub symmetric: bool,
}

/// Sharding metadata for multi-file models
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShardingMetadata {
    /// Total number of shards
    pub shard_count: usize,
    /// This shard's index (0-based)
    pub shard_index: usize,
    /// Total size across all shards
    pub total_size: u64,
    /// Shard file pattern (e.g., "model-{:05d}-of-{:05d}.apr")
    pub pattern: Option<String>,
}

// ============================================================================
// Tensor Index
// ============================================================================

/// Tensor index entry (fixed size for efficient lookup)
#[derive(Debug, Clone)]
pub struct TensorIndexEntry {
    /// Tensor name (up to 256 bytes)
    pub name: String,
    /// Data type
    pub dtype: TensorDType,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Offset in data section (64-byte aligned)
    pub offset: u64,
    /// Size in bytes
    pub size: u64,
}
