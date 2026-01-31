//! GGUF Import/Export (spec §7.2)
//!
//! Pure Rust reader/writer for GGUF format (llama.cpp compatible).
//! WASM compatible - no C/C++ dependencies.
//!
//! # Format Structure
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │ Magic: "GGUF" (4 bytes)                 │
//! │ Version: u32 (currently 3)              │
//! │ Tensor count: u64                       │
//! │ Metadata KV count: u64                  │
//! ├─────────────────────────────────────────┤
//! │ Metadata KV pairs                       │
//! ├─────────────────────────────────────────┤
//! │ Tensor info array                       │
//! ├─────────────────────────────────────────┤
//! │ Tensor data (aligned)                   │
//! └─────────────────────────────────────────┘
//! ```
//!
//! Reference: Gerganov, G. (2023). GGUF Format.

// Submodules (PMAT-199: split from monolithic gguf.rs)
pub mod types;
pub mod reader;
pub mod dequant;
pub mod api;

// Re-exports for backward compatibility
pub use types::*;
pub use reader::*;
pub use dequant::*;
pub use api::*;

// Module-level imports for test access via `use super::*;`
#[allow(unused_imports)]
use std::collections::BTreeMap;
#[allow(unused_imports)]
use std::io::{self, Read, Write};
#[allow(unused_imports)]
use std::fs::File;
#[allow(unused_imports)]
use std::path::Path;
#[allow(unused_imports)]
use crate::error::{AprenderError, Result};

#[cfg(test)]
mod tests {
    use super::*;

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

    fn make_test_reader(metadata: BTreeMap<String, GgufValue>) -> GgufReader {
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
        let reader = make_test_reader(BTreeMap::new());
        assert!(reader.vocabulary().is_none());
    }

    #[test]
    fn test_reader_vocabulary_present() {
        let mut metadata = BTreeMap::new();
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
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::ArrayString(vec![]),
        );
        let reader = make_test_reader(metadata);
        assert!(reader.vocabulary().is_none());
    }

    #[test]
    fn test_reader_tokenizer_model_none() {
        let reader = make_test_reader(BTreeMap::new());
        assert!(reader.tokenizer_model().is_none());
    }

    #[test]
    fn test_reader_tokenizer_model_present() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            GgufValue::String("llama".into()),
        );
        let reader = make_test_reader(metadata);
        assert_eq!(reader.tokenizer_model(), Some("llama".into()));
    }

    #[test]
    fn test_reader_bos_token_id_none() {
        let reader = make_test_reader(BTreeMap::new());
        assert!(reader.bos_token_id().is_none());
    }

    #[test]
    fn test_reader_bos_token_id_present() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.bos_token_id".to_string(),
            GgufValue::Uint32(1),
        );
        let reader = make_test_reader(metadata);
        assert_eq!(reader.bos_token_id(), Some(1));
    }

    #[test]
    fn test_reader_eos_token_id_none() {
        let reader = make_test_reader(BTreeMap::new());
        assert!(reader.eos_token_id().is_none());
    }

    #[test]
    fn test_reader_eos_token_id_present() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            GgufValue::Uint32(2),
        );
        let reader = make_test_reader(metadata);
        assert_eq!(reader.eos_token_id(), Some(2));
    }

    #[test]
    fn test_reader_architecture_none() {
        let reader = make_test_reader(BTreeMap::new());
        assert!(reader.architecture().is_none());
    }

    #[test]
    fn test_reader_architecture_present() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("qwen2".into()),
        );
        let reader = make_test_reader(metadata);
        assert_eq!(reader.architecture(), Some("qwen2".into()));
    }

    #[test]
    fn test_reader_model_name_none() {
        let reader = make_test_reader(BTreeMap::new());
        assert!(reader.model_name().is_none());
    }

    #[test]
    fn test_reader_model_name_present() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.name".to_string(),
            GgufValue::String("My Model".into()),
        );
        let reader = make_test_reader(metadata);
        assert_eq!(reader.model_name(), Some("My Model".into()));
    }

    #[test]
    fn test_reader_hidden_size_none() {
        let reader = make_test_reader(BTreeMap::new());
        assert!(reader.hidden_size().is_none());
    }

    #[test]
    fn test_reader_hidden_size_uint32() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "llama.embedding_length".to_string(),
            GgufValue::Uint32(4096),
        );
        let reader = make_test_reader(metadata);
        assert_eq!(reader.hidden_size(), Some(4096));
    }

    #[test]
    fn test_reader_hidden_size_uint64() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "llama.embedding_length".to_string(),
            GgufValue::Uint64(4096),
        );
        let reader = make_test_reader(metadata);
        assert_eq!(reader.hidden_size(), Some(4096));
    }

    #[test]
    fn test_reader_num_layers_none() {
        let reader = make_test_reader(BTreeMap::new());
        assert!(reader.num_layers().is_none());
    }

    #[test]
    fn test_reader_num_layers_uint32() {
        let mut metadata = BTreeMap::new();
        metadata.insert("llama.block_count".to_string(), GgufValue::Uint32(32));
        let reader = make_test_reader(metadata);
        assert_eq!(reader.num_layers(), Some(32));
    }

    #[test]
    fn test_reader_num_layers_uint64() {
        let mut metadata = BTreeMap::new();
        metadata.insert("llama.block_count".to_string(), GgufValue::Uint64(32));
        let reader = make_test_reader(metadata);
        assert_eq!(reader.num_layers(), Some(32));
    }

    #[test]
    fn test_reader_num_heads_none() {
        let reader = make_test_reader(BTreeMap::new());
        assert!(reader.num_heads().is_none());
    }

    #[test]
    fn test_reader_num_heads_uint32() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "llama.attention.head_count".to_string(),
            GgufValue::Uint32(32),
        );
        let reader = make_test_reader(metadata);
        assert_eq!(reader.num_heads(), Some(32));
    }

    #[test]
    fn test_reader_num_heads_uint64() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "llama.attention.head_count".to_string(),
            GgufValue::Uint64(32),
        );
        let reader = make_test_reader(metadata);
        assert_eq!(reader.num_heads(), Some(32));
    }

    #[test]
    fn test_reader_num_kv_heads_none_fallback_to_num_heads() {
        let mut metadata = BTreeMap::new();
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
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "llama.attention.head_count_kv".to_string(),
            GgufValue::Uint32(8),
        );
        let reader = make_test_reader(metadata);
        assert_eq!(reader.num_kv_heads(), Some(8));
    }

    #[test]
    fn test_reader_num_kv_heads_uint64() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "llama.attention.head_count_kv".to_string(),
            GgufValue::Uint64(8),
        );
        let reader = make_test_reader(metadata);
        assert_eq!(reader.num_kv_heads(), Some(8));
    }

    #[test]
    fn test_reader_vocab_size_none() {
        let reader = make_test_reader(BTreeMap::new());
        assert!(reader.vocab_size().is_none());
    }

    #[test]
    fn test_reader_vocab_size_uint32() {
        let mut metadata = BTreeMap::new();
        metadata.insert("llama.vocab_size".to_string(), GgufValue::Uint32(32000));
        let reader = make_test_reader(metadata);
        assert_eq!(reader.vocab_size(), Some(32000));
    }

    #[test]
    fn test_reader_vocab_size_uint64() {
        let mut metadata = BTreeMap::new();
        metadata.insert("llama.vocab_size".to_string(), GgufValue::Uint64(32000));
        let reader = make_test_reader(metadata);
        assert_eq!(reader.vocab_size(), Some(32000));
    }

    #[test]
    fn test_reader_vocab_size_from_vocabulary() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::ArrayString(vec!["a".into(), "b".into(), "c".into()]),
        );
        let reader = make_test_reader(metadata);
        assert_eq!(reader.vocab_size(), Some(3));
    }

    #[test]
    fn test_reader_intermediate_size_none() {
        let reader = make_test_reader(BTreeMap::new());
        assert!(reader.intermediate_size().is_none());
    }

    #[test]
    fn test_reader_intermediate_size_uint32() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "llama.feed_forward_length".to_string(),
            GgufValue::Uint32(11008),
        );
        let reader = make_test_reader(metadata);
        assert_eq!(reader.intermediate_size(), Some(11008));
    }

    #[test]
    fn test_reader_intermediate_size_uint64() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "llama.feed_forward_length".to_string(),
            GgufValue::Uint64(11008),
        );
        let reader = make_test_reader(metadata);
        assert_eq!(reader.intermediate_size(), Some(11008));
    }

    #[test]
    fn test_reader_context_length_none() {
        let reader = make_test_reader(BTreeMap::new());
        assert!(reader.context_length().is_none());
    }

    #[test]
    fn test_reader_context_length_uint32() {
        let mut metadata = BTreeMap::new();
        metadata.insert("llama.context_length".to_string(), GgufValue::Uint32(4096));
        let reader = make_test_reader(metadata);
        assert_eq!(reader.context_length(), Some(4096));
    }

    #[test]
    fn test_reader_context_length_uint64() {
        let mut metadata = BTreeMap::new();
        metadata.insert("llama.context_length".to_string(), GgufValue::Uint64(4096));
        let reader = make_test_reader(metadata);
        assert_eq!(reader.context_length(), Some(4096));
    }

    #[test]
    fn test_reader_rope_theta_none() {
        let reader = make_test_reader(BTreeMap::new());
        assert!(reader.rope_theta().is_none());
    }

    #[test]
    fn test_reader_rope_theta_float32() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "llama.rope.freq_base".to_string(),
            GgufValue::Float32(10000.0),
        );
        let reader = make_test_reader(metadata);
        assert_eq!(reader.rope_theta(), Some(10000.0));
    }

    #[test]
    fn test_reader_rope_theta_uint32() {
        let mut metadata = BTreeMap::new();
        metadata.insert("llama.rope.freq_base".to_string(), GgufValue::Uint32(10000));
        let reader = make_test_reader(metadata);
        assert_eq!(reader.rope_theta(), Some(10000.0));
    }

    #[test]
    fn test_reader_rms_norm_eps_none() {
        let reader = make_test_reader(BTreeMap::new());
        assert!(reader.rms_norm_eps().is_none());
    }

    #[test]
    fn test_reader_rms_norm_eps_float32() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "llama.attention.layer_norm_rms_epsilon".to_string(),
            GgufValue::Float32(1e-6),
        );
        let reader = make_test_reader(metadata);
        assert!((reader.rms_norm_eps().unwrap() - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn test_reader_with_custom_architecture() {
        let mut metadata = BTreeMap::new();
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
            tensors: BTreeMap::new(),
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
            tensors: BTreeMap::new(),
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

    // ========================================================================
    // read_u32/read_u64/read_string tests (Coverage)
    // ========================================================================

    #[test]
    fn test_read_u32_success() {
        let data = 0x12345678u32.to_le_bytes();
        let result = read_u32(&data, 0).expect("should read u32");
        assert_eq!(result, 0x12345678);
    }

    #[test]
    fn test_read_u32_eof() {
        let data = [1, 2, 3]; // Only 3 bytes
        let result = read_u32(&data, 0);
        assert!(result.is_err());
        assert!(format!("{:?}", result.unwrap_err()).contains("EOF"));
    }

    #[test]
    fn test_read_u32_offset() {
        let data = [0, 0, 0, 0, 0x78, 0x56, 0x34, 0x12];
        let result = read_u32(&data, 4).expect("should read u32 at offset");
        assert_eq!(result, 0x12345678);
    }

    #[test]
    fn test_read_u64_success() {
        let data = 0x123456789ABCDEF0u64.to_le_bytes();
        let result = read_u64(&data, 0).expect("should read u64");
        assert_eq!(result, 0x123456789ABCDEF0);
    }

    #[test]
    fn test_read_u64_eof() {
        let data = [1, 2, 3, 4, 5, 6, 7]; // Only 7 bytes
        let result = read_u64(&data, 0);
        assert!(result.is_err());
        assert!(format!("{:?}", result.unwrap_err()).contains("EOF"));
    }

    #[test]
    fn test_read_string_success() {
        // Length (8 bytes) + "hello" (5 bytes)
        let mut data = Vec::new();
        data.extend_from_slice(&5u64.to_le_bytes());
        data.extend_from_slice(b"hello");
        let (s, consumed) = read_string(&data, 0).expect("should read string");
        assert_eq!(s, "hello");
        assert_eq!(consumed, 13); // 8 + 5
    }

    #[test]
    fn test_read_string_eof() {
        // Length says 100 but only 5 bytes
        let mut data = Vec::new();
        data.extend_from_slice(&100u64.to_le_bytes());
        data.extend_from_slice(b"hello");
        let result = read_string(&data, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_string_empty() {
        let mut data = Vec::new();
        data.extend_from_slice(&0u64.to_le_bytes());
        let (s, consumed) = read_string(&data, 0).expect("should read empty string");
        assert_eq!(s, "");
        assert_eq!(consumed, 8);
    }

    // ========================================================================
    // read_metadata_value tests (Coverage for all type branches)
    // ========================================================================

    #[test]
    fn test_read_metadata_value_uint8() {
        let data = [42u8];
        let (value, consumed) = read_metadata_value(&data, 0, 0).expect("type 0 = uint8");
        assert!(matches!(value, GgufValue::Uint8(42)));
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_read_metadata_value_uint8_eof() {
        let data: [u8; 0] = [];
        let result = read_metadata_value(&data, 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_metadata_value_int8() {
        let data = [0xFE]; // -2 as i8
        let (value, consumed) = read_metadata_value(&data, 0, 1).expect("type 1 = int8");
        assert!(matches!(value, GgufValue::Int8(-2)));
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_read_metadata_value_int8_eof() {
        let data: [u8; 0] = [];
        let result = read_metadata_value(&data, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_metadata_value_uint16() {
        let data = 1000u16.to_le_bytes();
        let (value, consumed) = read_metadata_value(&data, 0, 2).expect("type 2 = uint16");
        assert!(matches!(value, GgufValue::Uint16(1000)));
        assert_eq!(consumed, 2);
    }

    #[test]
    fn test_read_metadata_value_uint16_eof() {
        let data = [1u8];
        let result = read_metadata_value(&data, 0, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_metadata_value_int16() {
        let data = (-1000i16).to_le_bytes();
        let (value, consumed) = read_metadata_value(&data, 0, 3).expect("type 3 = int16");
        assert!(matches!(value, GgufValue::Int16(-1000)));
        assert_eq!(consumed, 2);
    }

    #[test]
    fn test_read_metadata_value_int16_eof() {
        let data = [1u8];
        let result = read_metadata_value(&data, 0, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_metadata_value_uint32() {
        let data = 100_000u32.to_le_bytes();
        let (value, consumed) = read_metadata_value(&data, 0, 4).expect("type 4 = uint32");
        assert!(matches!(value, GgufValue::Uint32(100_000)));
        assert_eq!(consumed, 4);
    }

    #[test]
    fn test_read_metadata_value_int32() {
        let data = (-100_000i32).to_le_bytes();
        let (value, consumed) = read_metadata_value(&data, 0, 5).expect("type 5 = int32");
        assert!(matches!(value, GgufValue::Int32(-100_000)));
        assert_eq!(consumed, 4);
    }

    #[test]
    fn test_read_metadata_value_int32_eof() {
        let data = [1, 2, 3];
        let result = read_metadata_value(&data, 0, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_metadata_value_float32() {
        let data = 3.14f32.to_le_bytes();
        let (value, consumed) = read_metadata_value(&data, 0, 6).expect("type 6 = float32");
        if let GgufValue::Float32(v) = value {
            assert!((v - 3.14).abs() < 0.001);
        } else {
            panic!("expected Float32");
        }
        assert_eq!(consumed, 4);
    }

    #[test]
    fn test_read_metadata_value_float32_eof() {
        let data = [1, 2, 3];
        let result = read_metadata_value(&data, 0, 6);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_metadata_value_bool_true() {
        let data = [1u8];
        let (value, consumed) = read_metadata_value(&data, 0, 7).expect("type 7 = bool");
        assert!(matches!(value, GgufValue::Bool(true)));
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_read_metadata_value_bool_false() {
        let data = [0u8];
        let (value, consumed) = read_metadata_value(&data, 0, 7).expect("type 7 = bool");
        assert!(matches!(value, GgufValue::Bool(false)));
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_read_metadata_value_bool_eof() {
        let data: [u8; 0] = [];
        let result = read_metadata_value(&data, 0, 7);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_metadata_value_string() {
        let mut data = Vec::new();
        data.extend_from_slice(&5u64.to_le_bytes());
        data.extend_from_slice(b"hello");
        let (value, consumed) = read_metadata_value(&data, 0, 8).expect("type 8 = string");
        assert!(matches!(value, GgufValue::String(s) if s == "hello"));
        assert_eq!(consumed, 13);
    }

    #[test]
    fn test_read_metadata_value_array_string() {
        // Array: element_type (4) + count (8) + elements
        let mut data = Vec::new();
        data.extend_from_slice(&8u32.to_le_bytes()); // element type 8 = string
        data.extend_from_slice(&2u64.to_le_bytes()); // 2 elements
                                                     // First string "ab"
        data.extend_from_slice(&2u64.to_le_bytes());
        data.extend_from_slice(b"ab");
        // Second string "cd"
        data.extend_from_slice(&2u64.to_le_bytes());
        data.extend_from_slice(b"cd");
        let (value, consumed) = read_metadata_value(&data, 0, 9).expect("type 9 = array");
        if let GgufValue::ArrayString(arr) = value {
            assert_eq!(arr, vec!["ab".to_string(), "cd".to_string()]);
        } else {
            panic!("expected ArrayString");
        }
        // 4 (type) + 8 (count) + 10 (s1) + 10 (s2) = 32
        assert_eq!(consumed, 32);
    }

    #[test]
    fn test_read_metadata_value_array_uint32() {
        let mut data = Vec::new();
        data.extend_from_slice(&4u32.to_le_bytes()); // element type 4 = uint32
        data.extend_from_slice(&3u64.to_le_bytes()); // 3 elements
        data.extend_from_slice(&10u32.to_le_bytes());
        data.extend_from_slice(&20u32.to_le_bytes());
        data.extend_from_slice(&30u32.to_le_bytes());
        let (value, consumed) = read_metadata_value(&data, 0, 9).expect("type 9 = array");
        if let GgufValue::ArrayUint32(arr) = value {
            assert_eq!(arr, vec![10, 20, 30]);
        } else {
            panic!("expected ArrayUint32");
        }
        // 4 (type) + 8 (count) + 12 (3 * 4) = 24
        assert_eq!(consumed, 24);
    }

    #[test]
    fn test_read_metadata_value_array_int32() {
        let mut data = Vec::new();
        data.extend_from_slice(&5u32.to_le_bytes()); // element type 5 = int32
        data.extend_from_slice(&2u64.to_le_bytes()); // 2 elements
        data.extend_from_slice(&(-10i32).to_le_bytes());
        data.extend_from_slice(&20i32.to_le_bytes());
        let (value, consumed) = read_metadata_value(&data, 0, 9).expect("type 9 = array");
        if let GgufValue::ArrayInt32(arr) = value {
            assert_eq!(arr, vec![-10, 20]);
        } else {
            panic!("expected ArrayInt32");
        }
        // 4 (type) + 8 (count) + 8 (2 * 4) = 20
        assert_eq!(consumed, 20);
    }

    #[test]
    fn test_read_metadata_value_array_float32() {
        let mut data = Vec::new();
        data.extend_from_slice(&6u32.to_le_bytes()); // element type 6 = float32
        data.extend_from_slice(&2u64.to_le_bytes()); // 2 elements
        data.extend_from_slice(&1.5f32.to_le_bytes());
        data.extend_from_slice(&2.5f32.to_le_bytes());
        let (value, consumed) = read_metadata_value(&data, 0, 9).expect("type 9 = array");
        if let GgufValue::ArrayFloat32(arr) = value {
            assert!((arr[0] - 1.5).abs() < 0.001);
            assert!((arr[1] - 2.5).abs() < 0.001);
        } else {
            panic!("expected ArrayFloat32");
        }
        // 4 (type) + 8 (count) + 8 (2 * 4) = 20
        assert_eq!(consumed, 20);
    }

    // ========================================================================
    // f16_to_f32 tests
    // ========================================================================

    #[test]
    fn test_f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn test_f16_to_f32_one() {
        // F16 1.0 = 0x3C00
        let result = f16_to_f32(0x3C00);
        assert!((result - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_negative_one() {
        // F16 -1.0 = 0xBC00
        let result = f16_to_f32(0xBC00);
        assert!((result + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_subnormal() {
        // Very small subnormal number
        let result = f16_to_f32(0x0001);
        assert!(result > 0.0 && result < 0.001);
    }

    // ========================================================================
    // Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_gguf_value_type_enum_values() {
        assert_eq!(GgufValueType::Uint8 as u32, 0);
        assert_eq!(GgufValueType::Int8 as u32, 1);
        assert_eq!(GgufValueType::Uint16 as u32, 2);
        assert_eq!(GgufValueType::Int16 as u32, 3);
        assert_eq!(GgufValueType::Uint32 as u32, 4);
        assert_eq!(GgufValueType::Int32 as u32, 5);
        assert_eq!(GgufValueType::Float32 as u32, 6);
        assert_eq!(GgufValueType::Bool as u32, 7);
        assert_eq!(GgufValueType::String as u32, 8);
        assert_eq!(GgufValueType::Array as u32, 9);
        assert_eq!(GgufValueType::Uint64 as u32, 10);
        assert_eq!(GgufValueType::Int64 as u32, 11);
        assert_eq!(GgufValueType::Float64 as u32, 12);
    }

    #[test]
    fn test_ggml_type_enum_values() {
        assert_eq!(GgmlType::F32 as u32, 0);
        assert_eq!(GgmlType::F16 as u32, 1);
        assert_eq!(GgmlType::Q4_0 as u32, 2);
        assert_eq!(GgmlType::Q4_1 as u32, 3);
        assert_eq!(GgmlType::Q8_0 as u32, 8);
        assert_eq!(GgmlType::I8 as u32, 24);
        assert_eq!(GgmlType::I16 as u32, 25);
        assert_eq!(GgmlType::I32 as u32, 26);
        assert_eq!(GgmlType::I64 as u32, 27);
        assert_eq!(GgmlType::F64 as u32, 28);
    }

    #[test]
    fn test_gguf_value_type_debug() {
        let t = GgufValueType::String;
        assert!(format!("{:?}", t).contains("String"));
    }

    #[test]
    fn test_ggml_type_debug() {
        let t = GgmlType::F32;
        assert!(format!("{:?}", t).contains("F32"));
    }

    #[test]
    fn test_gguf_value_type_clone_copy() {
        let t1 = GgufValueType::Uint32;
        let t2 = t1; // Copy
        let t3 = t1.clone();
        assert_eq!(t1, t2);
        assert_eq!(t1, t3);
    }

    #[test]
    fn test_ggml_type_clone_copy() {
        let t1 = GgmlType::F16;
        let t2 = t1;
        let t3 = t1.clone();
        assert_eq!(t1, t2);
        assert_eq!(t1, t3);
    }

    #[test]
    fn test_gguf_header_debug() {
        let header = GgufHeader {
            version: GGUF_VERSION,
            tensor_count: 10,
            metadata_kv_count: 5,
        };
        assert!(format!("{:?}", header).contains("GgufHeader"));
    }

    #[test]
    fn test_gguf_tensor_info_debug() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            n_dims: 2,
            dims: vec![10, 20],
            dtype: GgmlType::F32,
            offset: 0,
        };
        assert!(format!("{:?}", info).contains("GgufTensorInfo"));
    }

    #[test]
    fn test_gguf_tensor_debug() {
        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: vec![10],
            dtype: GgmlType::F32,
            data: vec![0u8; 40],
        };
        assert!(format!("{:?}", tensor).contains("GgufTensor"));
    }

    #[test]
    fn test_gguf_value_debug() {
        let v = GgufValue::Uint32(42);
        assert!(format!("{:?}", v).contains("Uint32"));
    }

    #[test]
    fn test_f16_to_f32_max() {
        // F16 max ~65504
        let result = f16_to_f32(0x7BFF);
        assert!((result - 65504.0).abs() < 100.0);
    }

    #[test]
    fn test_f16_to_f32_negative_zero() {
        // Negative zero in f16 = 0x8000
        let result = f16_to_f32(0x8000);
        assert!((result - 0.0).abs() < 0.001 || result == -0.0);
    }

    #[test]
    fn test_f16_to_f32_half() {
        // F16 0.5 = 0x3800
        let result = f16_to_f32(0x3800);
        assert!((result - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_two() {
        // F16 2.0 = 0x4000
        let result = f16_to_f32(0x4000);
        assert!((result - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_gguf_tensor_clone() {
        let tensor = GgufTensor {
            name: "test".to_string(),
            shape: vec![10],
            dtype: GgmlType::F32,
            data: vec![1, 2, 3, 4],
        };
        let cloned = tensor.clone();
        assert_eq!(tensor.name, cloned.name);
        assert_eq!(tensor.data, cloned.data);
    }

    #[test]
    fn test_gguf_tensor_info_clone() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            n_dims: 2,
            dims: vec![10, 20],
            dtype: GgmlType::F32,
            offset: 100,
        };
        let cloned = info.clone();
        assert_eq!(info.name, cloned.name);
        assert_eq!(info.offset, cloned.offset);
    }

    #[test]
    fn test_gguf_value_clone_string() {
        let v = GgufValue::String("hello".to_string());
        let cloned = v.clone();
        if let (GgufValue::String(s1), GgufValue::String(s2)) = (&v, &cloned) {
            assert_eq!(s1, s2);
        } else {
            panic!("Expected String values");
        }
    }

    #[test]
    fn test_gguf_header_clone_full() {
        let header = GgufHeader {
            version: 3,
            tensor_count: 5,
            metadata_kv_count: 10,
        };
        let cloned = header.clone();
        assert_eq!(header.version, cloned.version);
    }

    #[test]
    fn test_padding_edge_cases() {
        assert_eq!(padding_for_alignment(0, 1), 0);
        assert_eq!(padding_for_alignment(1, 1), 0);
        assert_eq!(padding_for_alignment(100, 1), 0);
        assert_eq!(padding_for_alignment(31, 32), 1);
        assert_eq!(padding_for_alignment(63, 64), 1);
    }

    #[test]
    fn test_export_empty_tensors() {
        let mut buffer = Vec::new();
        let tensors: Vec<GgufTensor> = vec![];
        let metadata: Vec<(String, GgufValue)> = vec![];
        export_tensors_to_gguf(&mut buffer, &tensors, &metadata).expect("export");
        assert!(!buffer.is_empty());
        assert_eq!(&buffer[0..4], b"GGUF");
    }

    #[test]
    fn test_export_single_tensor() {
        let mut buffer = Vec::new();
        let tensor = GgufTensor {
            name: "test.weight".to_string(),
            shape: vec![4, 4],
            dtype: GgmlType::F32,
            data: vec![0u8; 64], // 16 floats * 4 bytes
        };
        export_tensors_to_gguf(&mut buffer, &[tensor], &[]).expect("export");
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_export_with_metadata() {
        let mut buffer = Vec::new();
        let metadata = vec![
            (
                "general.name".to_string(),
                GgufValue::String("test".to_string()),
            ),
            ("general.version".to_string(), GgufValue::Uint32(1)),
        ];
        export_tensors_to_gguf(&mut buffer, &[], &metadata).expect("export");
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_gguf_tensor_info_write_1d() {
        let info = GgufTensorInfo {
            name: "t".to_string(),
            n_dims: 1,
            dims: vec![100],
            dtype: GgmlType::F32,
            offset: 0,
        };
        let mut buffer = Vec::new();
        info.write_to(&mut buffer).expect("write");
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_gguf_tensor_info_write_4d() {
        let info = GgufTensorInfo {
            name: "tensor".to_string(),
            n_dims: 4,
            dims: vec![10, 20, 30, 40],
            dtype: GgmlType::F16,
            offset: 1024,
        };
        let mut buffer = Vec::new();
        info.write_to(&mut buffer).expect("write");
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_constants() {
        assert_eq!(GGUF_VERSION, 3);
        assert_eq!(GGUF_DEFAULT_ALIGNMENT, 32);
    }

    #[test]
    fn test_ggml_type_i8() {
        let t = GgmlType::I8;
        assert_eq!(t as u32, 24);
    }

    #[test]
    fn test_ggml_type_i16() {
        let t = GgmlType::I16;
        assert_eq!(t as u32, 25);
    }

    #[test]
    fn test_ggml_type_i32() {
        let t = GgmlType::I32;
        assert_eq!(t as u32, 26);
    }

    #[test]
    fn test_ggml_type_i64() {
        let t = GgmlType::I64;
        assert_eq!(t as u32, 27);
    }

    #[test]
    fn test_ggml_type_f64() {
        let t = GgmlType::F64;
        assert_eq!(t as u32, 28);
    }

    // ========================================================================
    // Dequantize Function Tests (ROSETTA-ML-001)
    // ========================================================================

    #[test]
    fn test_dequantize_q4_0_basic() {
        // Q4_0 block: 2 bytes d (f16) + 16 bytes qs = 18 bytes for 32 elements
        // Create a minimal valid block
        let mut data = Vec::new();
        // d = 1.0 in f16 (0x3C00)
        data.extend_from_slice(&0x3C00u16.to_le_bytes());
        // 16 quantized bytes (4-bit pairs) = 32 values
        data.extend_from_slice(&[0x00u8; 16]); // All zeros -> output should be around -8*d

        let result = dequantize_q4_0(&data, 0, 32).unwrap();
        assert_eq!(result.len(), 32);
        // Q4_0 subtracts 8 from each 4-bit value, so 0 becomes -8
        // All values should be -8 * d = -8.0
        for &v in &result {
            assert!((v + 8.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_dequantize_q4_0_out_of_bounds() {
        let data = vec![0u8; 10]; // Too small for one block
        let result = dequantize_q4_0(&data, 0, 32);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q8_0_basic() {
        // Q8_0 block: 2 bytes d (f16) + 32 bytes qs = 34 bytes for 32 elements
        let mut data = Vec::new();
        // d = 1.0 in f16 (0x3C00)
        data.extend_from_slice(&0x3C00u16.to_le_bytes());
        // 32 quantized bytes (signed i8) - all zeros
        data.extend_from_slice(&[0i8 as u8; 32]);

        let result = dequantize_q8_0(&data, 0, 32).unwrap();
        assert_eq!(result.len(), 32);
        // All values should be 0 * d = 0.0
        for &v in &result {
            assert!((v - 0.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_dequantize_q8_0_with_values() {
        // Q8_0 block: 2 bytes d (f16) + 32 bytes qs = 34 bytes for 32 elements
        let mut data = Vec::new();
        // d = 0.5 in f16 (0x3800)
        data.extend_from_slice(&0x3800u16.to_le_bytes());
        // First value = 10, rest = 0
        let mut qs = [0u8; 32];
        qs[0] = 10;
        data.extend_from_slice(&qs);

        let result = dequantize_q8_0(&data, 0, 32).unwrap();
        assert_eq!(result.len(), 32);
        // First value should be 10 * 0.5 = 5.0
        assert!((result[0] - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_dequantize_q8_0_out_of_bounds() {
        let data = vec![0u8; 20]; // Too small for one block (needs 34)
        let result = dequantize_q8_0(&data, 0, 32);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q5_0_basic() {
        // Q5_0 block size: 2 (d) + 4 (qh) + 16 (ql) = 22 bytes for 32 elements
        let mut data = Vec::new();
        // d = 1.0 in f16 (0x3C00)
        data.extend_from_slice(&0x3C00u16.to_le_bytes());
        // qh (high bits): 4 bytes
        data.extend_from_slice(&[0u8; 4]);
        // ql (low 4 bits): 16 bytes
        data.extend_from_slice(&[0u8; 16]);

        let result = dequantize_q5_0(&data, 0, 32).unwrap();
        assert_eq!(result.len(), 32);
        // Values are finite
        for &v in &result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_dequantize_q5_0_out_of_bounds() {
        let data = vec![0u8; 10]; // Too small (needs 22)
        let result = dequantize_q5_0(&data, 0, 32);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q5_1_basic() {
        // Q5_1 block size: 2 (d) + 2 (m) + 4 (qh) + 16 (ql) = 24 bytes for 32 elements
        let mut data = Vec::new();
        // d = 1.0 in f16 (0x3C00)
        data.extend_from_slice(&0x3C00u16.to_le_bytes());
        // m = 0.0 in f16 (0x0000)
        data.extend_from_slice(&0x0000u16.to_le_bytes());
        // qh (high bits): 4 bytes
        data.extend_from_slice(&[0u8; 4]);
        // ql (low 4 bits): 16 bytes
        data.extend_from_slice(&[0u8; 16]);

        let result = dequantize_q5_1(&data, 0, 32).unwrap();
        assert_eq!(result.len(), 32);
        // Values are finite
        for &v in &result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_dequantize_q5_1_out_of_bounds() {
        let data = vec![0u8; 10]; // Too small (needs 24)
        let result = dequantize_q5_1(&data, 0, 32);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q4_1_basic() {
        // Q4_1 block: 2 (d) + 2 (m) + 16 (qs) = 20 bytes for 32 elements
        let mut data = Vec::new();
        // d = 1.0 in f16 (0x3C00)
        data.extend_from_slice(&0x3C00u16.to_le_bytes());
        // m = 0.0 in f16 (0x0000)
        data.extend_from_slice(&0x0000u16.to_le_bytes());
        // 16 quantized bytes
        data.extend_from_slice(&[0u8; 16]);

        let result = dequantize_q4_1(&data, 0, 32).unwrap();
        assert_eq!(result.len(), 32);
        // Values are finite
        for &v in &result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_dequantize_q4_1_out_of_bounds() {
        let data = vec![0u8; 10]; // Too small (needs 20)
        let result = dequantize_q4_1(&data, 0, 32);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q4_k_basic() {
        // Q4_K super-block: 144 bytes for 256 elements
        // Structure: 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs) = 144
        let data = vec![0u8; 144];

        let result = dequantize_q4_k(&data, 0, 256).unwrap();
        assert_eq!(result.len(), 256);
        // All zeros in data -> all zeros in output
        for &v in &result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_dequantize_q4_k_out_of_bounds() {
        let data = vec![0u8; 100]; // Too small (needs 144)
        let result = dequantize_q4_k(&data, 0, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q5_k_basic() {
        // Q5_K super-block: 176 bytes for 256 elements
        let data = vec![0u8; 176];

        let result = dequantize_q5_k(&data, 0, 256).unwrap();
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_dequantize_q5_k_out_of_bounds() {
        let data = vec![0u8; 100]; // Too small (needs 176)
        let result = dequantize_q5_k(&data, 0, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q6_k_basic() {
        // Q6_K super-block: 210 bytes for 256 elements
        let data = vec![0u8; 210];

        let result = dequantize_q6_k(&data, 0, 256).unwrap();
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_dequantize_q6_k_out_of_bounds() {
        let data = vec![0u8; 100]; // Too small (needs 210)
        let result = dequantize_q6_k(&data, 0, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q2_k_basic() {
        // Q2_K super-block: 84 bytes for 256 elements
        let data = vec![0u8; 84];

        let result = dequantize_q2_k(&data, 0, 256).unwrap();
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_dequantize_q2_k_out_of_bounds() {
        let data = vec![0u8; 50]; // Too small (needs 84)
        let result = dequantize_q2_k(&data, 0, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q3_k_basic() {
        // Q3_K super-block: 110 bytes for 256 elements
        let data = vec![0u8; 110];

        let result = dequantize_q3_k(&data, 0, 256).unwrap();
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_dequantize_q3_k_out_of_bounds() {
        let data = vec![0u8; 50]; // Too small (needs 110)
        let result = dequantize_q3_k(&data, 0, 256);
        assert!(result.is_err());
    }
}

// ============================================================================
// Property-Based Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // Strategy for generating valid GGUF headers
    fn arb_header() -> impl Strategy<Value = GgufHeader> {
        (0u64..1000, 0u64..100).prop_map(|(tensor_count, metadata_kv_count)| GgufHeader {
            version: GGUF_VERSION,
            tensor_count,
            metadata_kv_count,
        })
    }

    proptest! {
        /// Property: Header write always produces exactly 24 bytes
        #[test]
        fn prop_header_size_always_24(header in arb_header()) {
            let mut buffer = Vec::new();
            header.write_to(&mut buffer).expect("write");
            prop_assert_eq!(buffer.len(), 24);
        }

        /// Property: Header always starts with GGUF magic
        #[test]
        fn prop_header_magic(header in arb_header()) {
            let mut buffer = Vec::new();
            header.write_to(&mut buffer).expect("write");
            prop_assert_eq!(&buffer[0..4], b"GGUF");
        }

        /// Property: Header version is always 3
        #[test]
        fn prop_header_version(header in arb_header()) {
            let mut buffer = Vec::new();
            header.write_to(&mut buffer).expect("write");
            let version = u32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]);
            prop_assert_eq!(version, GGUF_VERSION);
        }

        /// Property: Padding is always less than alignment
        #[test]
        fn prop_padding_less_than_alignment(offset in 0usize..10000, alignment in 1usize..256) {
            let padding = padding_for_alignment(offset, alignment);
            prop_assert!(padding < alignment);
        }

        /// Property: offset + padding is always aligned
        #[test]
        fn prop_padded_offset_aligned(offset in 0usize..10000, alignment in 1usize..256) {
            let padding = padding_for_alignment(offset, alignment);
            prop_assert_eq!((offset + padding) % alignment, 0);
        }

        /// Property: Aligned offsets need zero padding
        #[test]
        fn prop_aligned_needs_no_padding(multiple in 0usize..1000, alignment in 1usize..256) {
            let offset = multiple * alignment;
            prop_assert_eq!(padding_for_alignment(offset, alignment), 0);
        }

        /// Property: String metadata key-value is non-empty
        #[test]
        fn prop_string_metadata_nonempty(
            key in "[a-z][a-z0-9_.]{0,30}",
            value in "[a-zA-Z0-9_ ]{0,100}"
        ) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, &key, &GgufValue::String(value)).expect("write");
            prop_assert!(!buffer.is_empty());
        }

        /// Property: Uint32 value roundtrip through bytes
        #[test]
        fn prop_uint32_value_written(value in any::<u32>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "test", &GgufValue::Uint32(value)).expect("write");
            // Key: 8 (len) + 4 (test) + type: 4 + value: 4 = 20 bytes
            prop_assert!(buffer.len() >= 20);
        }

        /// Property: Float32 value roundtrip through bytes
        #[test]
        fn prop_float32_value_written(value in any::<f32>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "f", &GgufValue::Float32(value)).expect("write");
            prop_assert!(!buffer.is_empty());
        }

        /// Property: Tensor export produces valid GGUF with magic
        #[test]
        fn prop_tensor_export_has_magic(
            name in "[a-z][a-z0-9.]{0,20}",
            dim0 in 1u64..100,
            dim1 in 1u64..100
        ) {
            let data = vec![0u8; (dim0 * dim1 * 4) as usize]; // f32 data
            let tensor = GgufTensor {
                name,
                shape: vec![dim0, dim1],
                dtype: GgmlType::F32,
                data,
            };
            let mut buffer = Vec::new();
            export_tensors_to_gguf(&mut buffer, &[tensor], &[]).expect("export");
            prop_assert_eq!(&buffer[0..4], b"GGUF");
        }

        // ================================================================
        // Metadata Roundtrip Property Tests
        // ================================================================

        /// Property: Bool true encodes to 1, false to 0
        #[test]
        fn prop_bool_value_encoding(value in any::<bool>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "b", &GgufValue::Bool(value)).expect("write");
            // Last byte is the bool value
            let last_byte = buffer[buffer.len() - 1];
            prop_assert_eq!(last_byte, u8::from(value));
        }

        /// Property: Int64 values encode correctly
        #[test]
        fn prop_int64_value_encoded(value in any::<i64>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "i", &GgufValue::Int64(value)).expect("write");
            // Last 8 bytes are the i64 value
            let bytes = &buffer[buffer.len() - 8..];
            let decoded = i64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
            ]);
            prop_assert_eq!(decoded, value);
        }

        /// Property: Uint64 values encode correctly
        #[test]
        fn prop_uint64_value_encoded(value in any::<u64>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "u", &GgufValue::Uint64(value)).expect("write");
            let bytes = &buffer[buffer.len() - 8..];
            let decoded = u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
            ]);
            prop_assert_eq!(decoded, value);
        }

        /// Property: Float64 values encode correctly (bit-exact)
        #[test]
        fn prop_float64_value_encoded(value in any::<f64>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "d", &GgufValue::Float64(value)).expect("write");
            let bytes = &buffer[buffer.len() - 8..];
            let decoded = f64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
            ]);
            // Use to_bits for NaN-safe comparison
            prop_assert_eq!(decoded.to_bits(), value.to_bits());
        }

        /// Property: Value type tag is correct for all value types
        #[test]
        fn prop_value_type_tag_uint32(value in any::<u32>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "x", &GgufValue::Uint32(value)).expect("write");
            // Type is at bytes 12-15 (after key: 8 byte len + 1 byte "x" = 9 bytes, padded to 9, then type)
            // Key: u64 length (8) + "x" (1) = 9 bytes, then u32 type
            let type_bytes = &buffer[9..13];
            let type_val = u32::from_le_bytes([type_bytes[0], type_bytes[1], type_bytes[2], type_bytes[3]]);
            prop_assert_eq!(type_val, GgufValueType::Uint32 as u32);
        }

        // ================================================================
        // Tensor Info Property Tests
        // ================================================================

        /// Property: Tensor info serialization contains name
        #[test]
        fn prop_tensor_info_contains_name(
            name in "[a-z][a-z0-9_.]{0,30}"
        ) {
            let info = GgufTensorInfo {
                name: name.clone(),
                n_dims: 2,
                dims: vec![10, 20],
                dtype: GgmlType::F32,
                offset: 0,
            };
            let mut buffer = Vec::new();
            info.write_to(&mut buffer).expect("write");
            // Name length is first 8 bytes
            let name_len = u64::from_le_bytes([
                buffer[0], buffer[1], buffer[2], buffer[3],
                buffer[4], buffer[5], buffer[6], buffer[7],
            ]) as usize;
            prop_assert_eq!(name_len, name.len());
            // Name bytes follow
            let name_bytes = &buffer[8..8 + name_len];
            prop_assert_eq!(name_bytes, name.as_bytes());
        }

        /// Property: Tensor info n_dims matches shape length
        #[test]
        fn prop_tensor_info_ndims_matches_shape(
            dims in proptest::collection::vec(1u64..100, 1..5)
        ) {
            let info = GgufTensorInfo {
                name: "t".to_string(),
                n_dims: dims.len() as u32,
                dims: dims.clone(),
                dtype: GgmlType::F32,
                offset: 0,
            };
            let mut buffer = Vec::new();
            info.write_to(&mut buffer).expect("write");
            // After name (8 + 1 = 9 bytes), n_dims is next 4 bytes
            let n_dims = u32::from_le_bytes([buffer[9], buffer[10], buffer[11], buffer[12]]);
            prop_assert_eq!(n_dims as usize, dims.len());
        }

        /// Property: Multiple metadata pairs produces correct count in header
        #[test]
        fn prop_export_metadata_count(
            count in 0usize..10
        ) {
            let metadata: Vec<(String, GgufValue)> = (0..count)
                .map(|i| (format!("key{i}"), GgufValue::Uint32(i as u32)))
                .collect();
            let mut buffer = Vec::new();
            export_tensors_to_gguf(&mut buffer, &[], &metadata).expect("export");
            // KV count is at bytes 16-23 (after magic 4, version 4, tensor_count 8)
            let kv_count = u64::from_le_bytes([
                buffer[16], buffer[17], buffer[18], buffer[19],
                buffer[20], buffer[21], buffer[22], buffer[23],
            ]);
            prop_assert_eq!(kv_count as usize, count);
        }

        /// Property: Tensor count in header matches tensors provided
        #[test]
        fn prop_export_tensor_count(
            count in 0usize..5
        ) {
            let tensors: Vec<GgufTensor> = (0..count)
                .map(|i| GgufTensor {
                    name: format!("t{i}"),
                    shape: vec![4],
                    dtype: GgmlType::F32,
                    data: vec![0u8; 16], // 4 f32s = 16 bytes
                })
                .collect();
            let mut buffer = Vec::new();
            export_tensors_to_gguf(&mut buffer, &tensors, &[]).expect("export");
            // Tensor count is at bytes 8-15
            let tensor_count = u64::from_le_bytes([
                buffer[8], buffer[9], buffer[10], buffer[11],
                buffer[12], buffer[13], buffer[14], buffer[15],
            ]);
            prop_assert_eq!(tensor_count as usize, count);
        }
    }
}
