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
