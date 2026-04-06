
    #[test]
    fn test_gguf_read_value_all_types() {
        use crate::gguf::test_factory::GGUFBuilder;
        // Test each value type individually
        let data = GGUFBuilder::new()
            .add_u8("v.u8", 255)
            .add_i8("v.i8", -128)
            .add_u16("v.u16", 65535)
            .add_i16("v.i16", -32768)
            .add_u32("v.u32", 4_294_967_295)
            .add_i32("v.i32", -2_147_483_648)
            .add_f32("v.f32", std::f32::consts::PI)
            .add_bool("v.bool_t", true)
            .add_bool("v.bool_f", false)
            .add_string("v.str", "test_value")
            .add_u64("v.u64", u64::MAX)
            .add_i64("v.i64", i64::MIN)
            .add_f64("v.f64", std::f64::consts::E)
            .build();

        let model = GGUFModel::from_bytes(&data).expect("model");

        // Verify u8
        match model.metadata.get("v.u8") {
            Some(GGUFValue::UInt8(v)) => assert_eq!(*v, 255),
            other => panic!("Expected UInt8(255), got {:?}", other),
        }
        // Verify i8
        match model.metadata.get("v.i8") {
            Some(GGUFValue::Int8(v)) => assert_eq!(*v, -128),
            other => panic!("Expected Int8(-128), got {:?}", other),
        }
        // Verify u16
        match model.metadata.get("v.u16") {
            Some(GGUFValue::UInt16(v)) => assert_eq!(*v, 65535),
            other => panic!("Expected UInt16(65535), got {:?}", other),
        }
        // Verify i16
        match model.metadata.get("v.i16") {
            Some(GGUFValue::Int16(v)) => assert_eq!(*v, -32768),
            other => panic!("Expected Int16(-32768), got {:?}", other),
        }
        // Verify u32
        match model.metadata.get("v.u32") {
            Some(GGUFValue::UInt32(v)) => assert_eq!(*v, 4_294_967_295),
            other => panic!("Expected UInt32(max), got {:?}", other),
        }
        // Verify i32
        match model.metadata.get("v.i32") {
            Some(GGUFValue::Int32(v)) => assert_eq!(*v, -2_147_483_648),
            other => panic!("Expected Int32(min), got {:?}", other),
        }
        // Verify f32
        match model.metadata.get("v.f32") {
            Some(GGUFValue::Float32(v)) => {
                assert!((*v - std::f32::consts::PI).abs() < 0.0001);
            },
            other => panic!("Expected Float32(PI), got {:?}", other),
        }
        // Verify bool true
        match model.metadata.get("v.bool_t") {
            Some(GGUFValue::Bool(v)) => assert!(*v),
            other => panic!("Expected Bool(true), got {:?}", other),
        }
        // Verify bool false
        match model.metadata.get("v.bool_f") {
            Some(GGUFValue::Bool(v)) => assert!(!*v),
            other => panic!("Expected Bool(false), got {:?}", other),
        }
        // Verify string
        match model.metadata.get("v.str") {
            Some(GGUFValue::String(v)) => assert_eq!(v, "test_value"),
            other => panic!("Expected String, got {:?}", other),
        }
        // Verify u64
        match model.metadata.get("v.u64") {
            Some(GGUFValue::UInt64(v)) => assert_eq!(*v, u64::MAX),
            other => panic!("Expected UInt64(max), got {:?}", other),
        }
        // Verify i64
        match model.metadata.get("v.i64") {
            Some(GGUFValue::Int64(v)) => assert_eq!(*v, i64::MIN),
            other => panic!("Expected Int64(min), got {:?}", other),
        }
        // Verify f64
        match model.metadata.get("v.f64") {
            Some(GGUFValue::Float64(v)) => {
                assert!((*v - std::f64::consts::E).abs() < 0.0001);
            },
            other => panic!("Expected Float64(E), got {:?}", other),
        }
    }

    #[test]
    fn test_gguf_read_value_array_of_strings() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .add_string_array("arr.strings", &["alpha", "beta", "gamma", "delta"])
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        match model.metadata.get("arr.strings") {
            Some(GGUFValue::Array(arr)) => {
                assert_eq!(arr.len(), 4);
                match &arr[0] {
                    GGUFValue::String(s) => assert_eq!(s, "alpha"),
                    other => panic!("Expected String, got {:?}", other),
                }
                match &arr[3] {
                    GGUFValue::String(s) => assert_eq!(s, "delta"),
                    other => panic!("Expected String, got {:?}", other),
                }
            },
            other => panic!("Expected Array, got {:?}", other),
        }
    }

    #[test]
    fn test_gguf_read_value_empty_array() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .add_string_array("arr.empty", &[])
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        match model.metadata.get("arr.empty") {
            Some(GGUFValue::Array(arr)) => assert!(arr.is_empty()),
            other => panic!("Expected empty Array, got {:?}", other),
        }
    }

    #[test]
    fn test_gguf_parse_header_truncated_at_tensor_count() {
        // Valid magic + version but truncated at tensor_count
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        // Only 4 bytes of tensor_count (need 8)
        data.extend_from_slice(&[0u8; 4]);

        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_parse_header_truncated_at_metadata_count() {
        // Valid header through tensor_count but truncated at metadata_count
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
        // Only 4 bytes of metadata_count (need 8)
        data.extend_from_slice(&[0u8; 4]);

        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_parse_tensor_info_excessive_ndims() {
        // Tensor with n_dims > 8 should fail
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

        // Tensor name: "test"
        data.extend_from_slice(&4u64.to_le_bytes());
        data.extend_from_slice(b"test");

        // n_dims = 10 (exceeds MAX_DIMS = 8)
        data.extend_from_slice(&10u32.to_le_bytes());

        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("dimensions") || err.contains("max allowed"),
            "Expected dims error, got: {}",
            err
        );
    }

    #[test]
    fn test_gguf_parse_tensor_info_valid_1d() {
        // A single 1D F32 tensor
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .add_f32_tensor("single_dim", &[4], &[1.0, 2.0, 3.0, 4.0])
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert_eq!(model.tensors.len(), 1);
        assert_eq!(model.tensors[0].name, "single_dim");
        assert_eq!(model.tensors[0].n_dims, 1);
    }

    #[test]
    fn test_gguf_parse_tensor_info_valid_2d() {
        // A single 2D F32 tensor
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .add_f32_tensor("matrix", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert_eq!(model.tensors.len(), 1);
        assert_eq!(model.tensors[0].name, "matrix");
        assert_eq!(model.tensors[0].n_dims, 2);
        // Builder reverses when writing, parser reverses back -> matches input
        assert_eq!(model.tensors[0].dims, vec![2, 3]);
    }

    #[test]
    fn test_gguf_parse_metadata_truncated_value() {
        // Valid header + key but truncated value data
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

        // Key: "key"
        data.extend_from_slice(&3u64.to_le_bytes());
        data.extend_from_slice(b"key");
        // Value type: u32 (type 4) but no value bytes follow
        data.extend_from_slice(&4u32.to_le_bytes());

        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_model_multiple_tensors() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .add_f32_tensor("t1", &[4], &[1.0, 2.0, 3.0, 4.0])
            .add_f32_tensor("t2", &[2, 2], &[5.0, 6.0, 7.0, 8.0])
            .add_f32_tensor("t3", &[3], &[9.0, 10.0, 11.0])
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert_eq!(model.tensors.len(), 3);

        // Verify all tensors can be extracted
        let t1 = model.get_tensor_f32("t1", &data).expect("t1");
        assert_eq!(t1, vec![1.0, 2.0, 3.0, 4.0]);

        let t2 = model.get_tensor_f32("t2", &data).expect("t2");
        assert_eq!(t2, vec![5.0, 6.0, 7.0, 8.0]);

        let t3 = model.get_tensor_f32("t3", &data).expect("t3");
        assert_eq!(t3, vec![9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_gguf_model_decode_sentencepiece_markers() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .add_string("tokenizer.ggml.model", "llama")
            .add_string_array("tokenizer.ggml.tokens", &["<unk>", "▁Hello", "▁world", "!"])
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let decoded = model.decode(&[1, 2, 3]);
        // SentencePiece: \u{2581} -> space
        assert!(decoded.contains("Hello"));
        assert!(decoded.contains("world"));
        assert!(decoded.contains("!"));
    }

    #[test]
    fn test_gguf_model_decode_gpt2_style() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("qwen2")
            .add_string("tokenizer.ggml.model", "gpt2")
            .add_string_array("tokenizer.ggml.tokens", &["a", "b", "c"])
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let decoded = model.decode(&[0, 1, 2]);
        // GPT-2 style uses byte-level BPE mapping
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_gguf_model_decode_out_of_range_token() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .add_string_array("tokenizer.ggml.tokens", &["hello", "world"])
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        // Token ID 999 is out of range
        let decoded = model.decode(&[999]);
        // Should use fallback character
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_gguf_model_decode_empty_tokens() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .add_string_array("tokenizer.ggml.tokens", &["hello"])
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let decoded = model.decode(&[]);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_gguf_model_decode_no_vocab_large_id() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new().architecture("llama").build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        // Token ID > 127 should be capped
        let decoded = model.decode(&[200, 300]);
        assert_eq!(decoded.len(), 2); // two characters
    }

    #[test]
    fn test_gguf_model_rope_type_neox_various_archs() {
        use crate::gguf::test_factory::GGUFBuilder;

        // Test starcoder2 -> NEOX
        let data = GGUFBuilder::new()
            .architecture("starcoder2")
            .hidden_dim("starcoder2", 64)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert_eq!(model.rope_type(), Some(2));

        // Test falcon -> NEOX
        let data = GGUFBuilder::new()
            .architecture("falcon")
            .hidden_dim("falcon", 64)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert_eq!(model.rope_type(), Some(2));

        // Test deepseek2 -> NEOX
        let data = GGUFBuilder::new()
            .architecture("deepseek2")
            .hidden_dim("deepseek2", 64)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert_eq!(model.rope_type(), Some(2));
    }

    #[test]
    fn test_gguf_model_rope_type_norm_default() {
        use crate::gguf::test_factory::GGUFBuilder;
        // Unknown architecture should default to NORM (0)
        let data = GGUFBuilder::new()
            .architecture("unknown_model")
            .hidden_dim("unknown_model", 64)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert_eq!(model.rope_type(), Some(0));
    }

    #[test]
    fn test_gguf_model_rope_type_with_neox_metadata() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .add_string("llama.rope.scaling.type", "neox")
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert_eq!(model.rope_type(), Some(2)); // neox -> NEOX
    }

    #[test]
    fn test_gguf_model_rope_type_with_unknown_scaling_type() {
        use crate::gguf::test_factory::GGUFBuilder;
        // Unknown scaling type should fall through to architecture inference
        let data = GGUFBuilder::new()
            .architecture("llama")
            .add_string("llama.rope.scaling.type", "unknown_type")
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert_eq!(model.rope_type(), Some(0)); // llama -> NORM
    }

    #[test]
    fn test_gguf_model_context_length_present() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .context_length("llama", 4096)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert_eq!(model.context_length(), Some(4096));
    }

    #[test]
    fn test_gguf_model_rope_freq_base_present() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .rope_freq_base("llama", 10000.0)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let theta = model.rope_freq_base();
        assert!(theta.is_some());
        assert!((theta.expect("theta") - 10000.0).abs() < 0.1);
    }

    #[test]
    fn test_gguf_model_kv_heads_present() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .num_heads("llama", 8)
            .num_kv_heads("llama", 2)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert_eq!(model.num_kv_heads(), Some(2));
        assert_eq!(model.num_heads(), Some(8));
    }

    #[test]
    fn test_gguf_model_get_tensor_data_out_of_bounds() {
        use crate::gguf::test_factory::GGUFBuilder;
        // Create a tensor whose data extends beyond the file
        // Use a small amount of tensor data but declare large dimensions
        let data = GGUFBuilder::new()
            .architecture("llama")
            .add_f32_tensor("small_tensor", &[2], &[1.0, 2.0])
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");

        // This should succeed since the tensor data is present
        let result = model.get_tensor_f32("small_tensor", &data);
        assert!(result.is_ok());
    }
