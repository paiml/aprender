
    #[test]
    fn test_split_gpt2_fused_qkv_raw_data_bytes_not_divisible_passthrough() {
        use crate::format::gguf::GgufRawTensor;

        let mut tensors = BTreeMap::new();
        // shape is fine but data bytes not divisible by 3
        tensors.insert(
            "model.layers.0.self_attn.c_attn.weight".to_string(),
            GgufRawTensor {
                data: vec![0u8; 25], // 25 not divisible by 3
                shape: vec![6, 4],
                dtype: 0,
            },
        );

        Architecture::split_gpt2_fused_qkv_raw(&mut tensors);
        assert_eq!(tensors.len(), 1);
    }

    #[test]
    fn test_split_gpt2_fused_qkv_raw_no_c_attn_noop() {
        use crate::format::gguf::GgufRawTensor;

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            GgufRawTensor {
                data: vec![0u8; 16],
                shape: vec![4, 4],
                dtype: 0,
            },
        );

        Architecture::split_gpt2_fused_qkv_raw(&mut tensors);
        assert_eq!(tensors.len(), 1);
    }

    #[test]
    fn test_split_gpt2_fused_qkv_raw_preserves_dtype() {
        use crate::format::gguf::GgufRawTensor;

        let mut tensors = BTreeMap::new();
        // Q4_K dtype=12
        tensors.insert(
            "model.layers.0.self_attn.c_attn.weight".to_string(),
            GgufRawTensor {
                data: vec![0u8; 36],
                shape: vec![6, 4],
                dtype: 12, // Q4_K
            },
        );

        Architecture::split_gpt2_fused_qkv_raw(&mut tensors);

        assert_eq!(tensors.len(), 3);
        for t in tensors.values() {
            assert_eq!(t.dtype, 12, "dtype should be preserved after split");
        }
    }

    #[test]
    fn test_split_gpt2_fused_qkv_raw_bias_non_1d_passthrough() {
        use crate::format::gguf::GgufRawTensor;

        let mut tensors = BTreeMap::new();
        // Bias with 2D shape — should be treated as weight path due to shape, but
        // since name says "bias", it enters bias path which checks shape.len()==1
        tensors.insert(
            "model.layers.0.self_attn.c_attn.bias".to_string(),
            GgufRawTensor {
                data: vec![0u8; 24],
                shape: vec![4, 6], // 2D — not valid for bias split
                dtype: 0,
            },
        );

        Architecture::split_gpt2_fused_qkv_raw(&mut tensors);
        assert_eq!(tensors.len(), 1);
        assert!(tensors.contains_key("model.layers.0.self_attn.c_attn.bias"));
    }
