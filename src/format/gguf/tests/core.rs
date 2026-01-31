//! Core GGUF Tests - Value types, write tests, accessor tests
//! PMAT-085: Split from tests.rs for file health (lines 1-1089)

use super::super::*;

#[test]
fn test_magic_constant() {
    assert_eq!(GGUF_MAGIC, 0x4655_4747);
    assert_eq!(&GGUF_MAGIC.to_le_bytes(), b"GGUF");
}

#[test]
fn test_header_size() {
    let mut buffer = Vec::new();
    let header = GgufHeader {
        version: GGUF_VERSION,
        tensor_count: 0,
        metadata_kv_count: 0,
    };
    header.write_to(&mut buffer).expect("write");
    // magic (4) + version (4) + tensor_count (8) + kv_count (8) = 24
    assert_eq!(buffer.len(), 24);
}

#[test]
fn test_padding_calculation() {
    assert_eq!(padding_for_alignment(0, 32), 0);
    assert_eq!(padding_for_alignment(1, 32), 31);
    assert_eq!(padding_for_alignment(32, 32), 0);
    assert_eq!(padding_for_alignment(33, 32), 31);
    assert_eq!(padding_for_alignment(64, 32), 0);
}

#[test]
fn test_value_types() {
    assert_eq!(GgufValue::Uint32(42).value_type(), GgufValueType::Uint32);
    assert_eq!(
        GgufValue::String("test".to_string()).value_type(),
        GgufValueType::String
    );
    assert_eq!(GgufValue::Bool(true).value_type(), GgufValueType::Bool);
}

// ========================================================================
// GgufValue type coverage tests
// ========================================================================

#[test]
fn test_all_value_types() {
    assert_eq!(GgufValue::Uint8(1).value_type(), GgufValueType::Uint8);
    assert_eq!(GgufValue::Int8(-1).value_type(), GgufValueType::Int8);
    assert_eq!(GgufValue::Uint16(1).value_type(), GgufValueType::Uint16);
    assert_eq!(GgufValue::Int16(-1).value_type(), GgufValueType::Int16);
    assert_eq!(GgufValue::Uint32(1).value_type(), GgufValueType::Uint32);
    assert_eq!(GgufValue::Int32(-1).value_type(), GgufValueType::Int32);
    assert_eq!(GgufValue::Float32(1.0).value_type(), GgufValueType::Float32);
    assert_eq!(GgufValue::Bool(true).value_type(), GgufValueType::Bool);
    assert_eq!(
        GgufValue::String("s".into()).value_type(),
        GgufValueType::String
    );
    assert_eq!(GgufValue::Uint64(1).value_type(), GgufValueType::Uint64);
    assert_eq!(GgufValue::Int64(-1).value_type(), GgufValueType::Int64);
    assert_eq!(GgufValue::Float64(1.0).value_type(), GgufValueType::Float64);
    assert_eq!(
        GgufValue::ArrayUint32(vec![1]).value_type(),
        GgufValueType::Array
    );
    assert_eq!(
        GgufValue::ArrayInt32(vec![1]).value_type(),
        GgufValueType::Array
    );
    assert_eq!(
        GgufValue::ArrayFloat32(vec![1.0]).value_type(),
        GgufValueType::Array
    );
    assert_eq!(
        GgufValue::ArrayString(vec!["s".into()]).value_type(),
        GgufValueType::Array
    );
}

// ========================================================================
// write_metadata_kv tests
// ========================================================================

#[test]
fn test_write_metadata_kv_uint8() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "test", &GgufValue::Uint8(42)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_int8() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "key", &GgufValue::Int8(-5)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_uint16() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::Uint16(1000)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_int16() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::Int16(-1000)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_uint32() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::Uint32(100_000)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_int32() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::Int32(-100_000)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_float32() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::Float32(3.14)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_bool() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::Bool(true)).unwrap();
    assert!(!buf.is_empty());

    let mut buf2 = Vec::new();
    write_metadata_kv(&mut buf2, "k", &GgufValue::Bool(false)).unwrap();
    assert!(!buf2.is_empty());
}

#[test]
fn test_write_metadata_kv_string() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::String("hello".into())).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_uint64() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::Uint64(1_000_000_000)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_int64() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::Int64(-1_000_000_000)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_float64() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::Float64(3.14159265359)).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_array_uint32() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::ArrayUint32(vec![1, 2, 3])).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_array_int32() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::ArrayInt32(vec![-1, 0, 1])).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_array_float32() {
    let mut buf = Vec::new();
    write_metadata_kv(&mut buf, "k", &GgufValue::ArrayFloat32(vec![1.0, 2.0, 3.0])).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_write_metadata_kv_array_string() {
    let mut buf = Vec::new();
    write_metadata_kv(
        &mut buf,
        "k",
        &GgufValue::ArrayString(vec!["a".into(), "b".into()]),
    )
    .unwrap();
    assert!(!buf.is_empty());
}

// ========================================================================
// GgufTensorInfo tests
// ========================================================================

#[test]
fn test_tensor_info_write() {
    let info = GgufTensorInfo {
        name: "test.weight".to_string(),
        n_dims: 2,
        dims: vec![10, 20],
        dtype: GgmlType::F32,
        offset: 0,
    };
    let mut buf = Vec::new();
    info.write_to(&mut buf).unwrap();
    assert!(!buf.is_empty());
}

#[test]
fn test_tensor_info_all_dtypes() {
    // Test write_to works for various dtypes
    let dtypes = [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::I8,
        GgmlType::I16,
        GgmlType::I32,
        GgmlType::I64,
        GgmlType::F64,
    ];
    for dtype in dtypes {
        let info = GgufTensorInfo {
            name: "w".to_string(),
            n_dims: 2,
            dims: vec![10, 20],
            dtype,
            offset: 0,
        };
        let mut buf = Vec::new();
        info.write_to(&mut buf).unwrap();
        assert!(!buf.is_empty());
    }
}

#[test]
fn test_ggml_type_all_variants() {
    // Test all GgmlType variants for Debug and Eq
    let types = [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q8_0,
        GgmlType::I8,
        GgmlType::I16,
        GgmlType::I32,
        GgmlType::I64,
        GgmlType::F64,
    ];
    for t in types {
        assert_eq!(t, t);
        assert!(!format!("{t:?}").is_empty());
    }
}

#[test]
fn test_gguf_value_type_all_variants() {
    // Test all GgufValueType variants
    let types = [
        GgufValueType::Uint8,
        GgufValueType::Int8,
        GgufValueType::Uint16,
        GgufValueType::Int16,
        GgufValueType::Uint32,
        GgufValueType::Int32,
        GgufValueType::Float32,
        GgufValueType::Bool,
        GgufValueType::String,
        GgufValueType::Array,
        GgufValueType::Uint64,
        GgufValueType::Int64,
        GgufValueType::Float64,
    ];
    for t in types {
        assert_eq!(t, t);
        assert!(!format!("{t:?}").is_empty());
    }
}

// ========================================================================
// Enum Debug/Clone tests
// ========================================================================

#[test]
fn test_gguf_value_type_enum() {
    let t = GgufValueType::Uint8;
    assert_eq!(t, GgufValueType::Uint8);
    let cloned = t;
    assert_eq!(t, cloned);
    assert!(format!("{t:?}").contains("Uint8"));
}

#[test]
fn test_ggml_type_enum() {
    let t = GgmlType::F32;
    assert_eq!(t, GgmlType::F32);
    let cloned = t;
    assert_eq!(t, cloned);
    assert!(format!("{t:?}").contains("F32"));
}

#[test]
fn test_gguf_value_clone() {
    let v = GgufValue::String("test".to_string());
    let cloned = v.clone();
    assert!(format!("{cloned:?}").contains("test"));
}

#[test]
fn test_gguf_header_clone() {
    let h = GgufHeader {
        version: 3,
        tensor_count: 10,
        metadata_kv_count: 5,
    };
    let cloned = h.clone();
    assert_eq!(cloned.version, 3);
    assert_eq!(cloned.tensor_count, 10);
    assert!(format!("{cloned:?}").contains("GgufHeader"));
}

// ========================================================================
// GgufTensor tests
// ========================================================================

#[test]
fn test_gguf_tensor_byte_size_f32() {
    let tensor = GgufTensor {
        name: "weights".to_string(),
        shape: vec![10, 20],
        dtype: GgmlType::F32,
        data: vec![0; 800], // 10 * 20 * 4
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_byte_size_f16() {
    let tensor = GgufTensor {
        name: "weights".to_string(),
        shape: vec![10, 20],
        dtype: GgmlType::F16,
        data: vec![0; 400], // 10 * 20 * 2
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_byte_size_i8() {
    let tensor = GgufTensor {
        name: "weights".to_string(),
        shape: vec![10, 20],
        dtype: GgmlType::I8,
        data: vec![0; 200], // 10 * 20 * 1
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_byte_size_i16() {
    let tensor = GgufTensor {
        name: "weights".to_string(),
        shape: vec![10, 20],
        dtype: GgmlType::I16,
        data: vec![0; 400],
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_byte_size_i32() {
    let tensor = GgufTensor {
        name: "weights".to_string(),
        shape: vec![10, 20],
        dtype: GgmlType::I32,
        data: vec![0; 800],
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_byte_size_i64() {
    let tensor = GgufTensor {
        name: "weights".to_string(),
        shape: vec![10, 20],
        dtype: GgmlType::I64,
        data: vec![0; 1600],
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_byte_size_f64() {
    let tensor = GgufTensor {
        name: "weights".to_string(),
        shape: vec![10, 20],
        dtype: GgmlType::F64,
        data: vec![0; 1600],
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_byte_size_q4_0() {
    let tensor = GgufTensor {
        name: "quantized".to_string(),
        shape: vec![64],
        dtype: GgmlType::Q4_0,
        data: vec![0; 100],
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_byte_size_q4_1() {
    let tensor = GgufTensor {
        name: "quantized".to_string(),
        shape: vec![64],
        dtype: GgmlType::Q4_1,
        data: vec![0; 100],
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_byte_size_q8_0() {
    let tensor = GgufTensor {
        name: "quantized".to_string(),
        shape: vec![64],
        dtype: GgmlType::Q8_0,
        data: vec![0; 100],
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_clone_debug() {
    let tensor = GgufTensor {
        name: "test".to_string(),
        shape: vec![10],
        dtype: GgmlType::F32,
        data: vec![1, 2, 3, 4],
    };
    let cloned = tensor.clone();
    assert_eq!(cloned.name, "test");
    assert!(format!("{cloned:?}").contains("GgufTensor"));
}

// ========================================================================
// export_tensors_to_gguf tests
// ========================================================================

#[test]
fn test_export_tensors_to_gguf_empty() {
    let mut buf = Vec::new();
    export_tensors_to_gguf(&mut buf, &[], &[]).expect("export should succeed");
    // Should have header at minimum
    assert!(buf.len() >= 24);
}

#[test]
fn test_export_tensors_to_gguf_with_metadata() {
    let mut buf = Vec::new();
    let metadata = vec![
        (
            "model.name".to_string(),
            GgufValue::String("test".to_string()),
        ),
        ("model.version".to_string(), GgufValue::Uint32(1)),
    ];
    export_tensors_to_gguf(&mut buf, &[], &metadata).expect("export should succeed");
    assert!(buf.len() > 24);
}

#[test]
fn test_export_tensors_to_gguf_with_tensors() {
    let mut buf = Vec::new();
    let tensors = vec![GgufTensor {
        name: "weights".to_string(),
        shape: vec![4],
        dtype: GgmlType::F32,
        data: vec![0; 16], // 4 * 4 bytes
    }];
    export_tensors_to_gguf(&mut buf, &tensors, &[]).expect("export should succeed");
    assert!(buf.len() > 24);
}

#[test]
fn test_export_tensors_to_gguf_full() {
    let mut buf = Vec::new();
    let tensors = vec![
        GgufTensor {
            name: "layer.0.weight".to_string(),
            shape: vec![10, 10],
            dtype: GgmlType::F32,
            data: vec![0; 400],
        },
        GgufTensor {
            name: "layer.0.bias".to_string(),
            shape: vec![10],
            dtype: GgmlType::F32,
            data: vec![0; 40],
        },
    ];
    let metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String("test".to_string()),
        ),
        (
            "general.quantization_version".to_string(),
            GgufValue::Uint32(2),
        ),
    ];
    export_tensors_to_gguf(&mut buf, &tensors, &metadata).expect("export should succeed");
    // Verify header magic
    assert_eq!(&buf[0..4], b"GGUF");
}

#[test]
fn test_gguf_tensor_info_clone_debug() {
    let info = GgufTensorInfo {
        name: "test".to_string(),
        n_dims: 2,
        dims: vec![10, 20],
        dtype: GgmlType::F32,
        offset: 0,
    };
    let cloned = info.clone();
    assert_eq!(cloned.name, "test");
    assert!(format!("{cloned:?}").contains("GgufTensorInfo"));
}

// ========================================================================
// GgufReader accessor tests
// ========================================================================

fn make_test_reader(metadata: std::collections::BTreeMap<String, GgufValue>) -> GgufReader {
    GgufReader {
        data: vec![],
        version: GGUF_VERSION,
        tensor_count: 0,
        tensors: vec![],
        data_offset: 0,
        metadata,
    }
}

#[test]
fn test_reader_vocabulary_empty() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.vocabulary().is_none());
}

#[test]
fn test_reader_vocabulary_present() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "tokenizer.ggml.tokens".to_string(),
        GgufValue::ArrayString(vec!["hello".into(), "world".into()]),
    );
    let reader = make_test_reader(metadata);
    let vocab = reader.vocabulary().expect("should have vocab");
    assert_eq!(vocab, vec!["hello", "world"]);
}

#[test]
fn test_reader_vocabulary_empty_array() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "tokenizer.ggml.tokens".to_string(),
        GgufValue::ArrayString(vec![]),
    );
    let reader = make_test_reader(metadata);
    assert!(reader.vocabulary().is_none());
}

#[test]
fn test_reader_tokenizer_model_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.tokenizer_model().is_none());
}

#[test]
fn test_reader_tokenizer_model_present() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "tokenizer.ggml.model".to_string(),
        GgufValue::String("llama".into()),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.tokenizer_model(), Some("llama".into()));
}

#[test]
fn test_reader_bos_token_id_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.bos_token_id().is_none());
}

#[test]
fn test_reader_bos_token_id_present() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "tokenizer.ggml.bos_token_id".to_string(),
        GgufValue::Uint32(1),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.bos_token_id(), Some(1));
}

#[test]
fn test_reader_eos_token_id_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.eos_token_id().is_none());
}

#[test]
fn test_reader_eos_token_id_present() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "tokenizer.ggml.eos_token_id".to_string(),
        GgufValue::Uint32(2),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.eos_token_id(), Some(2));
}

#[test]
fn test_reader_architecture_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.architecture().is_none());
}

#[test]
fn test_reader_architecture_present() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String("qwen2".into()),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.architecture(), Some("qwen2".into()));
}

#[test]
fn test_reader_model_name_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.model_name().is_none());
}

#[test]
fn test_reader_model_name_present() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "general.name".to_string(),
        GgufValue::String("My Model".into()),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.model_name(), Some("My Model".into()));
}

#[test]
fn test_reader_hidden_size_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.hidden_size().is_none());
}

#[test]
fn test_reader_hidden_size_uint32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.embedding_length".to_string(),
        GgufValue::Uint32(4096),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.hidden_size(), Some(4096));
}

#[test]
fn test_reader_hidden_size_uint64() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.embedding_length".to_string(),
        GgufValue::Uint64(4096),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.hidden_size(), Some(4096));
}

#[test]
fn test_reader_num_layers_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.num_layers().is_none());
}

#[test]
fn test_reader_num_layers_uint32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert("llama.block_count".to_string(), GgufValue::Uint32(32));
    let reader = make_test_reader(metadata);
    assert_eq!(reader.num_layers(), Some(32));
}

#[test]
fn test_reader_num_layers_uint64() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert("llama.block_count".to_string(), GgufValue::Uint64(32));
    let reader = make_test_reader(metadata);
    assert_eq!(reader.num_layers(), Some(32));
}

#[test]
fn test_reader_num_heads_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.num_heads().is_none());
}

#[test]
fn test_reader_num_heads_uint32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.attention.head_count".to_string(),
        GgufValue::Uint32(32),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.num_heads(), Some(32));
}

#[test]
fn test_reader_num_heads_uint64() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.attention.head_count".to_string(),
        GgufValue::Uint64(32),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.num_heads(), Some(32));
}

#[test]
fn test_reader_num_kv_heads_none_fallback_to_num_heads() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.attention.head_count".to_string(),
        GgufValue::Uint32(32),
    );
    let reader = make_test_reader(metadata);
    // Without head_count_kv, should fall back to num_heads
    assert_eq!(reader.num_kv_heads(), Some(32));
}

#[test]
fn test_reader_num_kv_heads_uint32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.attention.head_count_kv".to_string(),
        GgufValue::Uint32(8),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.num_kv_heads(), Some(8));
}

#[test]
fn test_reader_num_kv_heads_uint64() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.attention.head_count_kv".to_string(),
        GgufValue::Uint64(8),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.num_kv_heads(), Some(8));
}

#[test]
fn test_reader_vocab_size_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.vocab_size().is_none());
}

#[test]
fn test_reader_vocab_size_uint32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert("llama.vocab_size".to_string(), GgufValue::Uint32(32000));
    let reader = make_test_reader(metadata);
    assert_eq!(reader.vocab_size(), Some(32000));
}

#[test]
fn test_reader_vocab_size_uint64() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert("llama.vocab_size".to_string(), GgufValue::Uint64(32000));
    let reader = make_test_reader(metadata);
    assert_eq!(reader.vocab_size(), Some(32000));
}

#[test]
fn test_reader_vocab_size_from_vocabulary() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "tokenizer.ggml.tokens".to_string(),
        GgufValue::ArrayString(vec!["a".into(), "b".into(), "c".into()]),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.vocab_size(), Some(3));
}

#[test]
fn test_reader_intermediate_size_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.intermediate_size().is_none());
}

#[test]
fn test_reader_intermediate_size_uint32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.feed_forward_length".to_string(),
        GgufValue::Uint32(11008),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.intermediate_size(), Some(11008));
}

#[test]
fn test_reader_intermediate_size_uint64() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.feed_forward_length".to_string(),
        GgufValue::Uint64(11008),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.intermediate_size(), Some(11008));
}

#[test]
fn test_reader_context_length_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.context_length().is_none());
}

#[test]
fn test_reader_context_length_uint32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert("llama.context_length".to_string(), GgufValue::Uint32(4096));
    let reader = make_test_reader(metadata);
    assert_eq!(reader.context_length(), Some(4096));
}

#[test]
fn test_reader_context_length_uint64() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert("llama.context_length".to_string(), GgufValue::Uint64(4096));
    let reader = make_test_reader(metadata);
    assert_eq!(reader.context_length(), Some(4096));
}

#[test]
fn test_reader_rope_theta_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.rope_theta().is_none());
}

#[test]
fn test_reader_rope_theta_float32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.rope.freq_base".to_string(),
        GgufValue::Float32(10000.0),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.rope_theta(), Some(10000.0));
}

#[test]
fn test_reader_rope_theta_uint32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert("llama.rope.freq_base".to_string(), GgufValue::Uint32(10000));
    let reader = make_test_reader(metadata);
    assert_eq!(reader.rope_theta(), Some(10000.0));
}

#[test]
fn test_reader_rms_norm_eps_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.rms_norm_eps().is_none());
}

#[test]
fn test_reader_rms_norm_eps_float32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.attention.layer_norm_rms_epsilon".to_string(),
        GgufValue::Float32(1e-6),
    );
    let reader = make_test_reader(metadata);
    assert!((reader.rms_norm_eps().unwrap() - 1e-6).abs() < 1e-12);
}

#[test]
fn test_reader_with_custom_architecture() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String("qwen2".into()),
    );
    metadata.insert(
        "qwen2.embedding_length".to_string(),
        GgufValue::Uint32(3584),
    );
    metadata.insert("qwen2.block_count".to_string(), GgufValue::Uint32(28));
    let reader = make_test_reader(metadata);
    assert_eq!(reader.architecture(), Some("qwen2".into()));
    assert_eq!(reader.hidden_size(), Some(3584));
    assert_eq!(reader.num_layers(), Some(28));
}

#[test]
fn test_gguf_tensor_meta_clone_debug() {
    let meta = GgufTensorMeta {
        name: "test.weight".to_string(),
        dims: vec![10, 20],
        dtype: 0, // F32
        offset: 0,
    };
    let cloned = meta.clone();
    assert_eq!(cloned.name, "test.weight");
    assert!(format!("{cloned:?}").contains("GgufTensorMeta"));
}

#[test]
fn test_gguf_tokenizer_has_vocabulary_false() {
    let tokenizer = GgufTokenizer {
        vocabulary: vec![],
        merges: vec![],
        model_type: None,
        bos_token_id: None,
        eos_token_id: None,
        architecture: None,
        model_name: None,
    };
    assert!(!tokenizer.has_vocabulary());
}

#[test]
fn test_gguf_tokenizer_has_vocabulary_true() {
    let tokenizer = GgufTokenizer {
        vocabulary: vec!["a".into()],
        merges: vec![],
        model_type: None,
        bos_token_id: None,
        eos_token_id: None,
        architecture: None,
        model_name: None,
    };
    assert!(tokenizer.has_vocabulary());
}

#[test]
fn test_gguf_tokenizer_vocab_size() {
    let tokenizer = GgufTokenizer {
        vocabulary: vec!["a".into(), "b".into(), "c".into()],
        merges: vec![],
        model_type: Some("llama".into()),
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        architecture: Some("llama".into()),
        model_name: Some("Test Model".into()),
    };
    assert_eq!(tokenizer.vocab_size(), 3);
    assert!(format!("{tokenizer:?}").contains("GgufTokenizer"));
}

#[test]
fn test_gguf_model_config_debug() {
    let config = GgufModelConfig {
        architecture: Some("llama".into()),
        hidden_size: Some(4096),
        num_layers: Some(32),
        num_heads: Some(32),
        num_kv_heads: Some(8),
        vocab_size: Some(32000),
        intermediate_size: Some(11008),
        max_position_embeddings: Some(4096),
        rope_theta: Some(10000.0),
        rms_norm_eps: Some(1e-6),
        rope_type: Some(0), // NORM style for LLaMA
    };
    assert!(format!("{config:?}").contains("GgufModelConfig"));
}

#[test]
fn test_gguf_load_result_debug() {
    let result = GgufLoadResult {
        tensors: std::collections::BTreeMap::new(),
        tokenizer: GgufTokenizer {
            vocabulary: vec![],
            merges: vec![],
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
            architecture: None,
            model_name: None,
        },
        model_config: GgufModelConfig {
            architecture: None,
            hidden_size: None,
            num_layers: None,
            num_heads: None,
            num_kv_heads: None,
            vocab_size: None,
            intermediate_size: None,
            max_position_embeddings: None,
            rope_theta: None,
            rms_norm_eps: None,
            rope_type: None,
        },
    };
    assert!(format!("{result:?}").contains("GgufLoadResult"));
}

#[test]
fn test_gguf_raw_tensor_debug() {
    let tensor = GgufRawTensor {
        data: vec![0u8; 16],
        shape: vec![4],
        dtype: 0, // F32
    };
    assert!(format!("{tensor:?}").contains("GgufRawTensor"));
}

#[test]
fn test_gguf_raw_load_result_debug() {
    let result = GgufRawLoadResult {
        tensors: std::collections::BTreeMap::new(),
        tokenizer: GgufTokenizer {
            vocabulary: vec![],
            merges: vec![],
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
            architecture: None,
            model_name: None,
        },
        model_config: GgufModelConfig {
            architecture: None,
            hidden_size: None,
            num_layers: None,
            num_heads: None,
            num_kv_heads: None,
            vocab_size: None,
            intermediate_size: None,
            max_position_embeddings: None,
            rope_theta: None,
            rms_norm_eps: None,
            rope_type: None,
        },
    };
    assert!(format!("{result:?}").contains("GgufRawLoadResult"));
}
