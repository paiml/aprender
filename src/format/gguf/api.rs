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
mod tests {
    use super::*;
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use tempfile::NamedTempFile;

    /// Helper to create a minimal GGUF file for testing
    fn create_test_gguf_file() -> NamedTempFile {
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");

        // Create minimal GGUF with one tensor and metadata
        let tensors = vec![GgufTensor {
            name: "test.weight".to_string(),
            shape: vec![4, 4],
            dtype: GgmlType::F32,
            data: vec![0u8; 64], // 16 floats * 4 bytes
        }];

        let metadata = vec![
            (
                "general.architecture".to_string(),
                GgufValue::String("llama".to_string()),
            ),
            (
                "general.name".to_string(),
                GgufValue::String("test-model".to_string()),
            ),
            ("llama.block_count".to_string(), GgufValue::Uint32(4)),
            ("llama.embedding_length".to_string(), GgufValue::Uint32(256)),
            (
                "llama.attention.head_count".to_string(),
                GgufValue::Uint32(4),
            ),
            (
                "llama.attention.head_count_kv".to_string(),
                GgufValue::Uint32(4),
            ),
            ("llama.vocab_size".to_string(), GgufValue::Uint32(32000)),
            (
                "llama.feed_forward_length".to_string(),
                GgufValue::Uint32(1024),
            ),
            ("llama.context_length".to_string(), GgufValue::Uint32(2048)),
            (
                "llama.rope.freq_base".to_string(),
                GgufValue::Float32(10000.0),
            ),
            (
                "llama.attention.layer_norm_rms_epsilon".to_string(),
                GgufValue::Float32(1e-6),
            ),
        ];

        let mut gguf_bytes = Vec::new();
        export_tensors_to_gguf(&mut gguf_bytes, &tensors, &metadata).expect("export GGUF");

        std::fs::write(file.path(), &gguf_bytes).expect("write GGUF");
        file
    }

    /// Helper to create GGUF with tokenizer data
    fn create_test_gguf_with_tokenizer() -> NamedTempFile {
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");

        let tensors = vec![GgufTensor {
            name: "token_embd.weight".to_string(),
            shape: vec![100, 64],
            dtype: GgmlType::F32,
            data: vec![0u8; 100 * 64 * 4],
        }];

        let metadata = vec![
            (
                "general.architecture".to_string(),
                GgufValue::String("llama".to_string()),
            ),
            (
                "general.name".to_string(),
                GgufValue::String("test-tokenizer-model".to_string()),
            ),
            (
                "tokenizer.ggml.model".to_string(),
                GgufValue::String("llama".to_string()),
            ),
            (
                "tokenizer.ggml.tokens".to_string(),
                GgufValue::ArrayString(vec![
                    "<unk>".to_string(),
                    "<s>".to_string(),
                    "</s>".to_string(),
                    "hello".to_string(),
                    "world".to_string(),
                ]),
            ),
            (
                "tokenizer.ggml.merges".to_string(),
                GgufValue::ArrayString(vec!["he llo".to_string(), "wor ld".to_string()]),
            ),
            (
                "tokenizer.ggml.bos_token_id".to_string(),
                GgufValue::Uint32(1),
            ),
            (
                "tokenizer.ggml.eos_token_id".to_string(),
                GgufValue::Uint32(2),
            ),
            ("llama.block_count".to_string(), GgufValue::Uint32(2)),
            ("llama.embedding_length".to_string(), GgufValue::Uint32(64)),
            (
                "llama.attention.head_count".to_string(),
                GgufValue::Uint32(2),
            ),
        ];

        let mut gguf_bytes = Vec::new();
        export_tensors_to_gguf(&mut gguf_bytes, &tensors, &metadata).expect("export GGUF");

        std::fs::write(file.path(), &gguf_bytes).expect("write GGUF");
        file
    }

    /// Helper to create GGUF with qwen2 architecture (for rope_type testing)
    fn create_test_gguf_qwen2() -> NamedTempFile {
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");

        let tensors = vec![GgufTensor {
            name: "model.embed_tokens.weight".to_string(),
            shape: vec![100, 64],
            dtype: GgmlType::F32,
            data: vec![0u8; 100 * 64 * 4],
        }];

        let metadata = vec![
            (
                "general.architecture".to_string(),
                GgufValue::String("qwen2".to_string()),
            ),
            (
                "general.name".to_string(),
                GgufValue::String("qwen2-test".to_string()),
            ),
            ("qwen2.block_count".to_string(), GgufValue::Uint32(4)),
            ("qwen2.embedding_length".to_string(), GgufValue::Uint32(64)),
            (
                "qwen2.attention.head_count".to_string(),
                GgufValue::Uint32(4),
            ),
        ];

        let mut gguf_bytes = Vec::new();
        export_tensors_to_gguf(&mut gguf_bytes, &tensors, &metadata).expect("export GGUF");

        std::fs::write(file.path(), &gguf_bytes).expect("write GGUF");
        file
    }

    // ========================================================================
    // load_gguf_tensors tests
    // ========================================================================

    #[test]
    fn test_load_gguf_tensors_success() {
        let file = create_test_gguf_file();
        let result = load_gguf_tensors(file.path());
        assert!(result.is_ok(), "load_gguf_tensors should succeed");

        let tensors = result.expect("tensors");
        assert_eq!(tensors.len(), 1);
        assert!(tensors.contains_key("test.weight"));

        let (data, shape) = &tensors["test.weight"];
        assert_eq!(shape, &[4, 4]);
        assert_eq!(data.len(), 16); // 4*4 f32 values
    }

    #[test]
    fn test_load_gguf_tensors_nonexistent_file() {
        let result = load_gguf_tensors("/nonexistent/path/model.gguf");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_gguf_tensors_invalid_file() {
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        std::fs::write(file.path(), b"not a valid gguf file").expect("write");

        let result = load_gguf_tensors(file.path());
        assert!(result.is_err());
    }

    // ========================================================================
    // load_gguf_with_tokenizer tests
    // ========================================================================

    #[test]
    fn test_load_gguf_with_tokenizer_success() {
        let file = create_test_gguf_with_tokenizer();
        let result = load_gguf_with_tokenizer(file.path());
        assert!(result.is_ok(), "load_gguf_with_tokenizer should succeed");

        let load_result = result.expect("load result");
        assert_eq!(load_result.tensors.len(), 1);

        // Check tokenizer
        let tokenizer = &load_result.tokenizer;
        assert!(tokenizer.has_vocabulary());
        assert_eq!(tokenizer.vocab_size(), 5);
        assert_eq!(tokenizer.vocabulary[0], "<unk>");
        assert_eq!(tokenizer.vocabulary[3], "hello");
        assert_eq!(tokenizer.bos_token_id, Some(1));
        assert_eq!(tokenizer.eos_token_id, Some(2));
        assert_eq!(tokenizer.model_type, Some("llama".to_string()));
        assert_eq!(tokenizer.merges.len(), 2);
        assert_eq!(tokenizer.architecture, Some("llama".to_string()));
        assert_eq!(
            tokenizer.model_name,
            Some("test-tokenizer-model".to_string())
        );
    }

    #[test]
    fn test_load_gguf_with_tokenizer_model_config() {
        let file = create_test_gguf_file();
        let result = load_gguf_with_tokenizer(file.path());
        assert!(result.is_ok());

        let load_result = result.expect("load result");
        let config = &load_result.model_config;

        assert_eq!(config.architecture, Some("llama".to_string()));
        assert_eq!(config.hidden_size, Some(256));
        assert_eq!(config.num_layers, Some(4));
        assert_eq!(config.num_heads, Some(4));
        assert_eq!(config.num_kv_heads, Some(4));
        assert_eq!(config.vocab_size, Some(32000));
        assert_eq!(config.intermediate_size, Some(1024));
        assert_eq!(config.max_position_embeddings, Some(2048));
        assert_eq!(config.rope_theta, Some(10000.0));
        assert!((config.rms_norm_eps.unwrap_or(0.0) - 1e-6).abs() < 1e-12);
        assert_eq!(config.rope_type, Some(0)); // NORM style for llama
    }

    #[test]
    fn test_load_gguf_with_tokenizer_qwen2_rope_type() {
        let file = create_test_gguf_qwen2();
        let result = load_gguf_with_tokenizer(file.path());
        assert!(result.is_ok());

        let load_result = result.expect("load result");
        let config = &load_result.model_config;

        assert_eq!(config.architecture, Some("qwen2".to_string()));
        // PMAT-114: Qwen2 should use NEOX-style RoPE (type 2)
        assert_eq!(config.rope_type, Some(2));
    }

    #[test]
    fn test_load_gguf_with_tokenizer_empty_vocab() {
        let file = create_test_gguf_file(); // No tokenizer metadata
        let result = load_gguf_with_tokenizer(file.path());
        assert!(result.is_ok());

        let load_result = result.expect("load result");
        let tokenizer = &load_result.tokenizer;

        assert!(!tokenizer.has_vocabulary());
        assert_eq!(tokenizer.vocab_size(), 0);
    }

    #[test]
    fn test_load_gguf_with_tokenizer_nonexistent() {
        let result = load_gguf_with_tokenizer("/nonexistent/model.gguf");
        assert!(result.is_err());
    }

    // ========================================================================
    // load_gguf_raw tests
    // ========================================================================

    #[test]
    fn test_load_gguf_raw_success() {
        let file = create_test_gguf_file();
        let result = load_gguf_raw(file.path());
        assert!(result.is_ok(), "load_gguf_raw should succeed");

        let load_result = result.expect("load result");
        assert_eq!(load_result.tensors.len(), 1);
        assert!(load_result.tensors.contains_key("test.weight"));

        let raw_tensor = &load_result.tensors["test.weight"];
        assert_eq!(raw_tensor.shape, vec![4, 4]);
        assert_eq!(raw_tensor.dtype, 0); // F32
        assert_eq!(raw_tensor.data.len(), 64); // 16 floats * 4 bytes
    }

    #[test]
    fn test_load_gguf_raw_preserves_quantization_dtype() {
        // Create GGUF with F16 tensor
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");

        let tensors = vec![GgufTensor {
            name: "quantized.weight".to_string(),
            shape: vec![8],
            dtype: GgmlType::F16,
            data: vec![0u8; 16], // 8 f16 values * 2 bytes
        }];

        let metadata = vec![(
            "general.architecture".to_string(),
            GgufValue::String("test".to_string()),
        )];

        let mut gguf_bytes = Vec::new();
        export_tensors_to_gguf(&mut gguf_bytes, &tensors, &metadata).expect("export GGUF");
        std::fs::write(file.path(), &gguf_bytes).expect("write GGUF");

        let result = load_gguf_raw(file.path());
        assert!(result.is_ok());

        let load_result = result.expect("load result");
        let raw_tensor = &load_result.tensors["quantized.weight"];
        assert_eq!(raw_tensor.dtype, 1); // F16
    }

    #[test]
    fn test_load_gguf_raw_model_config() {
        let file = create_test_gguf_file();
        let result = load_gguf_raw(file.path());
        assert!(result.is_ok());

        let load_result = result.expect("load result");
        let config = &load_result.model_config;

        assert_eq!(config.architecture, Some("llama".to_string()));
        assert_eq!(config.num_layers, Some(4));
        assert_eq!(config.rope_type, Some(0)); // NORM style
    }

    #[test]
    fn test_load_gguf_raw_qwen2_rope_type() {
        let file = create_test_gguf_qwen2();
        let result = load_gguf_raw(file.path());
        assert!(result.is_ok());

        let load_result = result.expect("load result");
        assert_eq!(load_result.model_config.rope_type, Some(2)); // NEOX style
    }

    #[test]
    fn test_load_gguf_raw_tokenizer_extraction() {
        let file = create_test_gguf_with_tokenizer();
        let result = load_gguf_raw(file.path());
        assert!(result.is_ok());

        let load_result = result.expect("load result");
        let tokenizer = &load_result.tokenizer;

        assert!(tokenizer.has_vocabulary());
        assert_eq!(tokenizer.vocab_size(), 5);
        assert_eq!(tokenizer.merges.len(), 2);
    }

    #[test]
    fn test_load_gguf_raw_nonexistent() {
        let result = load_gguf_raw("/nonexistent/model.gguf");
        assert!(result.is_err());
    }

    // ========================================================================
    // GgufTokenizer tests
    // ========================================================================

    #[test]
    fn test_gguf_tokenizer_default() {
        let tokenizer = GgufTokenizer::default();
        assert!(!tokenizer.has_vocabulary());
        assert_eq!(tokenizer.vocab_size(), 0);
        assert!(tokenizer.vocabulary.is_empty());
        assert!(tokenizer.merges.is_empty());
        assert!(tokenizer.model_type.is_none());
        assert!(tokenizer.bos_token_id.is_none());
        assert!(tokenizer.eos_token_id.is_none());
        assert!(tokenizer.architecture.is_none());
        assert!(tokenizer.model_name.is_none());
    }

    #[test]
    fn test_gguf_tokenizer_clone() {
        let tokenizer = GgufTokenizer {
            vocabulary: vec!["a".to_string(), "b".to_string()],
            merges: vec!["a b".to_string()],
            model_type: Some("llama".to_string()),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            architecture: Some("llama".to_string()),
            model_name: Some("test".to_string()),
            ..Default::default()
        };

        let cloned = tokenizer.clone();
        assert_eq!(cloned.vocabulary, tokenizer.vocabulary);
        assert_eq!(cloned.merges, tokenizer.merges);
        assert_eq!(cloned.model_type, tokenizer.model_type);
        assert_eq!(cloned.bos_token_id, tokenizer.bos_token_id);
        assert_eq!(cloned.eos_token_id, tokenizer.eos_token_id);
    }

    // ========================================================================
    // GgufModelConfig tests
    // ========================================================================

    #[test]
    fn test_gguf_model_config_default() {
        let config = GgufModelConfig::default();
        assert!(config.architecture.is_none());
        assert!(config.hidden_size.is_none());
        assert!(config.num_layers.is_none());
        assert!(config.num_heads.is_none());
        assert!(config.num_kv_heads.is_none());
        assert!(config.vocab_size.is_none());
        assert!(config.intermediate_size.is_none());
        assert!(config.max_position_embeddings.is_none());
        assert!(config.rope_theta.is_none());
        assert!(config.rms_norm_eps.is_none());
        assert!(config.rope_type.is_none());
    }

    #[test]
    fn test_gguf_model_config_clone() {
        let config = GgufModelConfig {
            architecture: Some("llama".to_string()),
            hidden_size: Some(4096),
            num_layers: Some(32),
            num_heads: Some(32),
            num_kv_heads: Some(8),
            vocab_size: Some(32000),
            intermediate_size: Some(11008),
            max_position_embeddings: Some(4096),
            rope_theta: Some(10000.0),
            rms_norm_eps: Some(1e-6),
            rope_type: Some(0),
        };

        let cloned = config.clone();
        assert_eq!(cloned.architecture, config.architecture);
        assert_eq!(cloned.hidden_size, config.hidden_size);
        assert_eq!(cloned.num_layers, config.num_layers);
        assert_eq!(cloned.rope_type, config.rope_type);
    }

    // ========================================================================
    // GgufRawTensor tests
    // ========================================================================

    #[test]
    fn test_gguf_raw_tensor_clone() {
        let tensor = GgufRawTensor {
            data: vec![1, 2, 3, 4],
            shape: vec![2, 2],
            dtype: 0,
        };

        let cloned = tensor.clone();
        assert_eq!(cloned.data, tensor.data);
        assert_eq!(cloned.shape, tensor.shape);
        assert_eq!(cloned.dtype, tensor.dtype);
    }

    // ========================================================================
    // Rope type inference tests (PMAT-114)
    // ========================================================================

    #[test]
    fn test_rope_type_inference_qwen_variants() {
        // Test all qwen variants get NEOX style
        for arch in ["qwen2", "qwen2.5", "qwen"] {
            let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");

            let tensors = vec![GgufTensor {
                name: "test.weight".to_string(),
                shape: vec![4],
                dtype: GgmlType::F32,
                data: vec![0u8; 16],
            }];

            let metadata = vec![(
                "general.architecture".to_string(),
                GgufValue::String(arch.to_string()),
            )];

            let mut gguf_bytes = Vec::new();
            export_tensors_to_gguf(&mut gguf_bytes, &tensors, &metadata).expect("export");
            std::fs::write(file.path(), &gguf_bytes).expect("write");

            let result = load_gguf_with_tokenizer(file.path()).expect("load");
            assert_eq!(
                result.model_config.rope_type,
                Some(2),
                "Architecture '{arch}' should use NEOX rope_type=2"
            );
        }
    }

    #[test]
    fn test_rope_type_inference_other_architectures() {
        // Test non-qwen architectures get NORM style
        for arch in ["llama", "mistral", "phi", "falcon", "gpt2"] {
            let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");

            let tensors = vec![GgufTensor {
                name: "test.weight".to_string(),
                shape: vec![4],
                dtype: GgmlType::F32,
                data: vec![0u8; 16],
            }];

            let metadata = vec![(
                "general.architecture".to_string(),
                GgufValue::String(arch.to_string()),
            )];

            let mut gguf_bytes = Vec::new();
            export_tensors_to_gguf(&mut gguf_bytes, &tensors, &metadata).expect("export");
            std::fs::write(file.path(), &gguf_bytes).expect("write");

            let result = load_gguf_with_tokenizer(file.path()).expect("load");
            assert_eq!(
                result.model_config.rope_type,
                Some(0),
                "Architecture '{arch}' should use NORM rope_type=0"
            );
        }
    }
}
