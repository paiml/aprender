//! v2 header impl + JSON metadata + tensor-index entry types (issue #2231).
//!
//! Formerly `include!`d into `v2/mod.rs`; now a real module. Reaches the
//! parent-scope header/flag/const definitions via `super::` and the sibling
//! [`super::TensorDType`] via the `v2` namespace re-export.

use super::{
    AprV2Flags, AprV2Header, TensorDType, V2FormatError, HEADER_SIZE_V2, MAGIC_V2, VERSION_V2,
};
use crate::crc32::crc32;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Model description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Model author/organization
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,

    /// Model license (SPDX identifier; governed by C-APR-PROVENANCE).
    /// NO skip_serializing_if here: FALSIFY-SHIP-022 requires provenance
    /// keys to serialize as explicit `null` (FM-APR-PROV-SILENT-SKIP) —
    /// and `license` is not a realizar alias-group member, so the null
    /// cannot trigger the duplicate-field poison (C-APR-MERGE-RUNNABLE).
    #[serde(default)]
    pub license: Option<String>,

    /// Training-data source (dataset identifier or "teacher-only";
    /// governed by C-APR-PROVENANCE / AC-SHIP2-012 / FALSIFY-SHIP-022).
    /// Explicit-null serialization required — see `license`.
    #[serde(default)]
    pub data_source: Option<String>,

    /// SPDX license for `data_source` (governed by C-APR-PROVENANCE /
    /// AC-SHIP2-012 / FALSIFY-SHIP-022).
    /// Explicit-null serialization required — see `license`.
    #[serde(default)]
    pub data_license: Option<String>,

    /// Model version string
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    /// Source/provenance URI (DD6: Model provenance tracking)
    /// Examples: "<hf://openai/whisper-tiny>", "<local://path/to/model.safetensors>"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// Original format before conversion
    /// Examples: "safetensors", "gguf", "pytorch"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_format: Option<String>,

    /// Creation timestamp (ISO 8601)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,

    /// Total model size in bytes
    #[serde(default)]
    pub total_size: u64,

    /// Parameter count
    #[serde(default)]
    pub param_count: u64,

    /// Quantization info
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<QuantizationMetadata>,

    /// Shard info (for multi-file models)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sharding: Option<ShardingMetadata>,

    /// Chat template (Jinja2 format, from tokenizer_config.json)
    /// Per spec: chat-template-improvement-spec.md CTA-01
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,

    /// Detected chat template format
    /// Per spec: chat-template-improvement-spec.md CTA-03
    /// Values: "chatml", "llama2", "mistral", "phi", "alpaca", "custom", "raw"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_format: Option<String>,

    /// Special tokens for chat templates
    /// Per spec: chat-template-improvement-spec.md CTA-04
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub special_tokens: Option<ChatSpecialTokens>,

    // ========================================================================
    // Transformer Config (CRITICAL for inference - realizar::apr::AprMetadata)
    // ========================================================================
    /// Model architecture family (e.g., "llama", "qwen2", "phi")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,

    /// HuggingFace class name from `config.json::architectures[0]`
    /// (e.g., "Qwen2ForCausalLM", "LlamaForCausalLM"). Distinct from
    /// `architecture` (family) and `model_type`. PMAT-690 P0-K stamps
    /// this so downstream `apr pretrain --init` can propagate it into
    /// the trained checkpoint's metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_architecture: Option<String>,

    /// HuggingFace `config.json::model_type` (e.g., "qwen2", "llama").
    /// PMAT-690 P0-K stamps this alongside `hf_architecture` so the
    /// import→pretrain→export chain has a single source of truth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_model_type: Option<String>,

    /// Hidden dimension size
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_size: Option<usize>,

    /// Number of transformer layers
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_layers: Option<usize>,

    /// Number of attention heads
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_heads: Option<usize>,

    /// Number of key-value heads (for GQA, defaults to num_heads)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_kv_heads: Option<usize>,

    /// Vocabulary size
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab_size: Option<usize>,

    /// FFN intermediate dimension
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intermediate_size: Option<usize>,

    /// Maximum context/sequence length
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_position_embeddings: Option<usize>,

    /// RoPE theta for position encoding
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_theta: Option<f32>,

    /// RoPE type: 0=NORM (adjacent pairs), 2=NEOX (split halves)
    /// CORRECTNESS-011: Qwen2.5 models require rope_type=2 (NEOX style)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_type: Option<u32>,

    /// Layer norm epsilon
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rms_norm_eps: Option<f32>,

    /// Explicit head dimension (overrides hidden_size / num_heads for Qwen3+)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_dim: Option<usize>,

    /// Number of MoE experts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_experts: Option<usize>,

    /// Number of experts selected per token
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_experts_per_tok: Option<usize>,

    /// MoE expert intermediate/FFN dimension
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe_intermediate_size: Option<usize>,

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

    /// Canonicalize HF/GGUF-style alias keys into typed struct fields
    /// (C-APR-MERGE-RUNNABLE / FALSIFY-APR-MERGE-RUNNABLE-001).
    ///
    /// Import-produced APR files carry HuggingFace-style dimension keys
    /// (`num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, …)
    /// which land in `custom` because this struct has no serde aliases.
    /// Realizar's `AprMetadata` deserializer DOES alias them — so a file
    /// containing BOTH the canonical field (even as an explicit `null`)
    /// AND an alias key makes serde fail with "duplicate field", which
    /// realizar's mmap loader swallows via `unwrap_or_default()` —
    /// silently dropping ALL metadata (architecture, dims, embedded
    /// tokenizer) and producing C-01 / "no tokenizer in APR metadata"
    /// failures on a file that physically contains everything.
    ///
    /// This method promotes alias values into the typed fields (when the
    /// field is unset) and REMOVES the alias keys from `custom`, so a
    /// re-serialized container has exactly one spelling per dimension.
    pub fn canonicalize_hf_aliases(&mut self) {
        fn take_usize(
            custom: &mut HashMap<String, serde_json::Value>,
            keys: &[&str],
        ) -> Option<usize> {
            let mut found = None;
            for k in keys {
                if let Some(v) = custom.remove(*k) {
                    if found.is_none() {
                        found = v.as_u64().and_then(|n| usize::try_from(n).ok());
                    }
                }
            }
            found
        }
        fn take_f32(custom: &mut HashMap<String, serde_json::Value>, keys: &[&str]) -> Option<f32> {
            let mut found = None;
            for k in keys {
                if let Some(v) = custom.remove(*k) {
                    if found.is_none() {
                        #[allow(clippy::cast_possible_truncation)]
                        {
                            found = v.as_f64().map(|n| n as f32);
                        }
                    }
                }
            }
            found
        }

        // Alias groups mirror realizar::apr::AprMetadata serde aliases (PMAT-111).
        let v = take_usize(&mut self.custom, &["hidden_dim", "d_model", "n_embd"]);
        self.hidden_size = self.hidden_size.or(v);
        let v = take_usize(
            &mut self.custom,
            &["num_hidden_layers", "n_layers", "n_layer"],
        );
        self.num_layers = self.num_layers.or(v);
        let v = take_usize(
            &mut self.custom,
            &["num_attention_heads", "n_heads", "n_head"],
        );
        self.num_heads = self.num_heads.or(v);
        let v = take_usize(&mut self.custom, &["num_key_value_heads", "n_kv_heads"]);
        self.num_kv_heads = self.num_kv_heads.or(v);
        let v = take_usize(&mut self.custom, &["n_vocab"]);
        self.vocab_size = self.vocab_size.or(v);
        let v = take_usize(
            &mut self.custom,
            &["ffn_dim", "intermediate_dim", "n_inner"],
        );
        self.intermediate_size = self.intermediate_size.or(v);
        let v = take_usize(
            &mut self.custom,
            &["max_seq_len", "context_length", "n_ctx"],
        );
        self.max_position_embeddings = self.max_position_embeddings.or(v);
        let v = take_f32(&mut self.custom, &["layer_norm_eps", "norm_eps"]);
        self.rms_norm_eps = self.rms_norm_eps.or(v);
    }

    /// Deserialize from JSON bytes
    ///
    /// # Errors
    /// Returns error if deserialization fails.
    pub fn from_json(data: &[u8]) -> Result<Self, V2FormatError> {
        // ALB-107: Parse as Value first to handle duplicate keys in metadata.
        // Entrenar checkpoints (v1-v9) may have duplicate fields like rms_norm_eps
        // due to #[serde(flatten)] serializing both struct field (null) and custom
        // map entry. Value::Object deduplicates (last value wins).
        let value: serde_json::Value = serde_json::from_slice(data)
            .map_err(|e| V2FormatError::MetadataError(e.to_string()))?;
        serde_json::from_value(value).map_err(|e| V2FormatError::MetadataError(e.to_string()))
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
