use super::*;

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
