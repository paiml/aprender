//! GGUF high-level API (load, tokenizer, model config)

use std::collections::BTreeMap;
use std::path::Path;

use crate::error::Result;

use super::{GgufReader, TensorDataMap};

/// Load GGUF file and extract all tensors as F32
pub fn load_gguf_tensors<P: AsRef<Path>>(path: P) -> Result<TensorDataMap> {
    let reader = GgufReader::from_file(path)?;
    reader.get_all_tensors_f32()
}

/// Tokenizer data extracted from GGUF file
#[derive(Debug, Clone, Default)]
pub struct GgufTokenizer {
    /// Vocabulary tokens (indexed by token ID)
    pub vocabulary: Vec<String>,
    /// BPE merge rules (PMAT-171) - "token1 token2" strings for encoding
    pub merges: Vec<String>,
    /// Tokenizer model type (e.g., "llama", "gpt2")
    pub model_type: Option<String>,
    /// BOS (beginning of sequence) token ID
    pub bos_token_id: Option<u32>,
    /// EOS (end of sequence) token ID
    pub eos_token_id: Option<u32>,
    /// Model architecture (e.g., "llama", "qwen2")
    pub architecture: Option<String>,
    /// Model name from metadata
    pub model_name: Option<String>,
    /// GH-253: Per-token type array (1=normal, 3=special, etc.)
    pub token_type: Vec<i32>,
    /// GH-253: Padding token ID
    pub padding_token_id: Option<u32>,
    /// GH-253: Whether to add BOS token
    pub add_bos_token: Option<bool>,
    /// GH-253: Chat template (Jinja2 format)
    pub chat_template: Option<String>,
    /// GH-277: Pre-tokenizer type (e.g., "default", "gpt-2", "qwen2")
    /// Preserved for GGUF→APR→GGUF round-trip fidelity
    pub pre_type: Option<String>,
}

impl GgufTokenizer {
    /// Check if tokenizer has a valid vocabulary
    #[must_use]
    pub fn has_vocabulary(&self) -> bool {
        !self.vocabulary.is_empty()
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }
}

/// Transformer model configuration extracted from GGUF metadata
/// CRITICAL for APR inference - must match realizar::apr::AprMetadata
#[derive(Debug, Clone, Default)]
pub struct GgufModelConfig {
    /// Model architecture family (e.g., "llama", "qwen2", "phi")
    pub architecture: Option<String>,
    /// Hidden dimension size (embedding_length)
    pub hidden_size: Option<usize>,
    /// Number of transformer layers (block_count)
    pub num_layers: Option<usize>,
    /// Number of attention heads
    pub num_heads: Option<usize>,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: Option<usize>,
    /// Vocabulary size
    pub vocab_size: Option<usize>,
    /// FFN intermediate dimension
    pub intermediate_size: Option<usize>,
    /// Maximum context length
    pub max_position_embeddings: Option<usize>,
    /// RoPE theta for position encoding
    pub rope_theta: Option<f32>,
    /// RMS norm epsilon
    pub rms_norm_eps: Option<f32>,
    /// RoPE type: 0=NORM (adjacent pairs), 2=NEOX (split halves)
    /// CORRECTNESS-011: Qwen2.5 models require rope_type=2 (NEOX style)
    pub rope_type: Option<u32>,
}

/// Result of loading a GGUF file with full tokenizer data and model config
#[derive(Debug)]
pub struct GgufLoadResult {
    /// Tensor data (name -> (data, shape))
    pub tensors: TensorDataMap,
    /// Tokenizer data extracted from GGUF metadata
    pub tokenizer: GgufTokenizer,
    /// Model configuration (CRITICAL for inference)
    pub model_config: GgufModelConfig,
}

/// Load GGUF file and extract tensors, tokenizer, AND model config
///
/// This is the preferred method for GGUF import as it preserves
/// the vocabulary and model config needed for text generation.
pub fn load_gguf_with_tokenizer<P: AsRef<Path>>(path: P) -> Result<GgufLoadResult> {
    let reader = GgufReader::from_file(path)?;

    let tensors = reader.get_all_tensors_f32()?;

    // PMAT-171: Extract both vocabulary and BPE merges for standalone APR encoding
    let tokenizer = GgufTokenizer {
        vocabulary: reader.vocabulary().unwrap_or_else(Vec::new),
        merges: reader.merges().unwrap_or_else(Vec::new),
        model_type: reader.tokenizer_model(),
        bos_token_id: reader.bos_token_id(),
        eos_token_id: reader.eos_token_id(),
        architecture: reader.architecture(),
        model_name: reader.model_name(),
        pre_type: reader.pre_tokenizer_type(),
        ..Default::default()
    };

    // PMAT-114: Infer rope_type from architecture
    // Qwen2/Qwen2.5 models use NEOX-style RoPE (type 2)
    let arch = reader.architecture();
    let rope_type = match arch.as_deref() {
        Some("qwen2" | "qwen2.5" | "qwen") => Some(2), // NEOX style
        _ => Some(0),                                  // Default to NORM style
    };

    let model_config = GgufModelConfig {
        architecture: arch,
        hidden_size: reader.hidden_size(),
        num_layers: reader.num_layers(),
        num_heads: reader.num_heads(),
        num_kv_heads: reader.num_kv_heads(),
        vocab_size: reader.vocab_size(),
        intermediate_size: reader.intermediate_size(),
        max_position_embeddings: reader.context_length(),
        rope_theta: reader.rope_theta(),
        rms_norm_eps: reader.rms_norm_eps(),
        rope_type,
    };

    Ok(GgufLoadResult {
        tensors,
        tokenizer,
        model_config,
    })
}

/// Raw tensor data with quantization preserved
#[derive(Debug, Clone)]
pub struct GgufRawTensor {
    /// Raw bytes (Q4K/Q6K super-blocks, or F32/F16 data)
    pub data: Vec<u8>,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// GGML dtype: 0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 8=Q8_0, 10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K
    pub dtype: u32,
}

/// Result of loading GGUF with raw quantized tensors
#[derive(Debug)]
pub struct GgufRawLoadResult {
    /// Tensors with raw bytes (preserving Q4K/Q6K quantization)
    pub tensors: BTreeMap<String, GgufRawTensor>,
    /// Extracted tokenizer
    pub tokenizer: GgufTokenizer,
    /// Model configuration
    pub model_config: GgufModelConfig,
}

/// Load GGUF with raw quantized tensors (preserves Q4K for GPU inference)
///
/// This is essential for APR format to achieve 2x Ollama performance.
/// The Q4K bytes are stored directly in APR and used by GPU kernels.
pub fn load_gguf_raw<P: AsRef<Path>>(path: P) -> Result<GgufRawLoadResult> {
    let reader = GgufReader::from_file(path)?;

    let raw_tensors = reader.get_all_tensors_raw()?;
    let mut tensors = BTreeMap::new();
    for (name, (data, shape, dtype)) in raw_tensors {
        tensors.insert(name, GgufRawTensor { data, shape, dtype });
    }

    // PMAT-171: Extract both vocabulary and BPE merges for standalone APR encoding
    // GH-253: Also extract token_type, padding_token_id, add_bos_token, chat_template
    // for GGUF→APR→GGUF round-trip fidelity
    let tokenizer = GgufTokenizer {
        vocabulary: reader.vocabulary().unwrap_or_else(Vec::new),
        merges: reader.merges().unwrap_or_else(Vec::new),
        model_type: reader.tokenizer_model(),
        bos_token_id: reader.bos_token_id(),
        eos_token_id: reader.eos_token_id(),
        architecture: reader.architecture(),
        model_name: reader.model_name(),
        token_type: reader.token_type().unwrap_or_default(),
        padding_token_id: reader.padding_token_id(),
        add_bos_token: reader.add_bos_token(),
        chat_template: reader.chat_template(),
        pre_type: reader.pre_tokenizer_type(),
    };

    // PMAT-114: Infer rope_type from architecture
    let arch = reader.architecture();
    let rope_type = match arch.as_deref() {
        Some("qwen2" | "qwen2.5" | "qwen") => Some(2), // NEOX style
        _ => Some(0),                                  // Default to NORM style
    };

    let model_config = GgufModelConfig {
        architecture: arch,
        hidden_size: reader.hidden_size(),
        num_layers: reader.num_layers(),
        num_heads: reader.num_heads(),
        num_kv_heads: reader.num_kv_heads(),
        vocab_size: reader.vocab_size(),
        intermediate_size: reader.intermediate_size(),
        max_position_embeddings: reader.context_length(),
        rope_theta: reader.rope_theta(),
        rms_norm_eps: reader.rms_norm_eps(),
        rope_type,
    };

    Ok(GgufRawLoadResult {
        tensors,
        tokenizer,
        model_config,
    })
}

#[cfg(test)]
#[path = "api_tests.rs"]
mod tests;
