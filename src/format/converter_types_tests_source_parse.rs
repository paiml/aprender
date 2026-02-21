
    #[test]
    fn test_hf_parse_resolve_main_stripped() {
        // GH-221: Users copy URLs with /resolve/main/ from HuggingFace browser
        let src = Source::parse("hf://Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors")
            .expect("should parse");
        match src {
            Source::HuggingFace { org, repo, file } => {
                assert_eq!(org, "Qwen");
                assert_eq!(repo, "Qwen2.5-1.5B-Instruct");
                assert_eq!(file, Some("model.safetensors".to_string()));
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    #[test]
    fn test_hf_parse_blob_main_stripped() {
        let src = Source::parse("hf://Qwen/Qwen2.5-1.5B-Instruct/blob/main/model.safetensors")
            .expect("should parse");
        match src {
            Source::HuggingFace { org, repo, file } => {
                assert_eq!(org, "Qwen");
                assert_eq!(repo, "Qwen2.5-1.5B-Instruct");
                assert_eq!(file, Some("model.safetensors".to_string()));
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    #[test]
    fn test_hf_parse_direct_file_unchanged() {
        let src = Source::parse("hf://Qwen/Qwen2.5-1.5B-Instruct/model.safetensors")
            .expect("should parse");
        match src {
            Source::HuggingFace { org, repo, file } => {
                assert_eq!(org, "Qwen");
                assert_eq!(repo, "Qwen2.5-1.5B-Instruct");
                assert_eq!(file, Some("model.safetensors".to_string()));
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    #[test]
    fn test_hf_parse_no_file() {
        let src = Source::parse("hf://Qwen/Qwen2.5-1.5B-Instruct").expect("should parse");
        match src {
            Source::HuggingFace { org, repo, file } => {
                assert_eq!(org, "Qwen");
                assert_eq!(repo, "Qwen2.5-1.5B-Instruct");
                assert_eq!(file, None);
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    #[test]
    fn test_hf_parse_nested_path_preserved() {
        // Paths that don't start with resolve/main/ or blob/main/ are preserved
        let src = Source::parse("hf://org/repo/subdir/model.safetensors").expect("should parse");
        match src {
            Source::HuggingFace { file, .. } => {
                assert_eq!(file, Some("subdir/model.safetensors".to_string()));
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    #[test]
    fn test_hf_parse_resolve_main_no_trailing_slash() {
        // Edge case: hf://org/repo/resolve/main (no file, no trailing slash)
        let src =
            Source::parse("hf://Qwen/Qwen2.5-1.5B-Instruct/resolve/main").expect("should parse");
        match src {
            Source::HuggingFace { org, repo, file } => {
                assert_eq!(org, "Qwen");
                assert_eq!(repo, "Qwen2.5-1.5B-Instruct");
                assert_eq!(file, None);
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    #[test]
    fn test_hf_parse_blob_main_no_trailing_slash() {
        let src = Source::parse("hf://Qwen/Qwen2.5-1.5B-Instruct/blob/main").expect("should parse");
        match src {
            Source::HuggingFace { file, .. } => {
                assert_eq!(file, None);
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    #[test]
    fn test_hf_parse_resolve_main_nested_file() {
        // resolve/main/ with nested subdir path
        let src =
            Source::parse("hf://Qwen/Qwen2.5-1.5B-Instruct/resolve/main/sub/model.safetensors")
                .expect("should parse");
        match src {
            Source::HuggingFace { file, .. } => {
                assert_eq!(file, Some("sub/model.safetensors".to_string()));
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    // ====================================================================
    // split_gpt2_fused_qkv: Coverage tests (impact 5.4)
    // ====================================================================

    #[test]
    fn test_split_gpt2_fused_qkv_bias_split_into_three() {
        let mut tensors = BTreeMap::new();
        // Bias: 1D tensor [3*hidden] where hidden=4
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        tensors.insert(
            "model.layers.0.self_attn.c_attn.bias".to_string(),
            (data, vec![12]),
        );

        Architecture::split_gpt2_fused_qkv(&mut tensors);

        assert_eq!(tensors.len(), 3);
        let (q_data, q_shape) = tensors.get("model.layers.0.self_attn.q_proj.bias").unwrap();
        assert_eq!(q_shape, &vec![4]);
        assert_eq!(q_data, &[0.0, 1.0, 2.0, 3.0]);

        let (k_data, k_shape) = tensors.get("model.layers.0.self_attn.k_proj.bias").unwrap();
        assert_eq!(k_shape, &vec![4]);
        assert_eq!(k_data, &[4.0, 5.0, 6.0, 7.0]);

        let (v_data, v_shape) = tensors.get("model.layers.0.self_attn.v_proj.bias").unwrap();
        assert_eq!(v_shape, &vec![4]);
        assert_eq!(v_data, &[8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_split_gpt2_fused_qkv_bias_not_divisible_by_3() {
        let mut tensors = BTreeMap::new();
        // Bias length not divisible by 3 -> should be kept as-is
        let data = vec![1.0f32; 7];
        tensors.insert(
            "model.layers.0.self_attn.c_attn.bias".to_string(),
            (data.clone(), vec![7]),
        );

        Architecture::split_gpt2_fused_qkv(&mut tensors);

        assert_eq!(tensors.len(), 1);
        assert!(tensors.contains_key("model.layers.0.self_attn.c_attn.bias"));
    }

    #[test]
    fn test_split_gpt2_fused_qkv_weight_safetensors_format() {
        let mut tensors = BTreeMap::new();
        // SafeTensors format: [hidden, 3*hidden] = [4, 12]
        // shape[1] == 3 * shape[0] → column split
        let hidden = 4;
        let total_cols = 3 * hidden;
        let mut data = vec![0.0f32; hidden * total_cols];
        for row in 0..hidden {
            for col in 0..total_cols {
                data[row * total_cols + col] = (row * total_cols + col) as f32;
            }
        }

        tensors.insert(
            "model.layers.0.self_attn.c_attn.weight".to_string(),
            (data, vec![hidden, total_cols]),
        );

        Architecture::split_gpt2_fused_qkv(&mut tensors);

        assert_eq!(tensors.len(), 3);

        let (q_data, q_shape) = tensors
            .get("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(q_shape, &vec![4, 4]); // [rows, cols_per_proj]
                                          // First row of Q: cols 0..4 from row 0 = [0, 1, 2, 3]
        assert_eq!(&q_data[0..4], &[0.0, 1.0, 2.0, 3.0]);

        let (k_data, k_shape) = tensors
            .get("model.layers.0.self_attn.k_proj.weight")
            .unwrap();
        assert_eq!(k_shape, &vec![4, 4]);
        // First row of K: cols 4..8 from row 0 = [4, 5, 6, 7]
        assert_eq!(&k_data[0..4], &[4.0, 5.0, 6.0, 7.0]);

        let (v_data, v_shape) = tensors
            .get("model.layers.0.self_attn.v_proj.weight")
            .unwrap();
        assert_eq!(v_shape, &vec![4, 4]);
        // First row of V: cols 8..12 from row 0 = [8, 9, 10, 11]
        assert_eq!(&v_data[0..4], &[8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_split_gpt2_fused_qkv_weight_gguf_format() {
        let mut tensors = BTreeMap::new();
        // GGUF format: [3*hidden, hidden] = [12, 4]
        // shape[0] % 3 == 0 → row split
        let hidden = 4;
        let total_rows = 3 * hidden;
        let mut data = vec![0.0f32; total_rows * hidden];
        for i in 0..data.len() {
            data[i] = i as f32;
        }

        tensors.insert(
            "model.layers.0.self_attn.c_attn.weight".to_string(),
            (data, vec![total_rows, hidden]),
        );

        Architecture::split_gpt2_fused_qkv(&mut tensors);

        assert_eq!(tensors.len(), 3);

        let (q_data, q_shape) = tensors
            .get("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(q_shape, &vec![4, 4]); // [rows_per_proj, cols]
        assert_eq!(q_data.len(), 16);
        // First chunk: elements 0..16
        assert_eq!(q_data[0], 0.0);

        let (k_data, k_shape) = tensors
            .get("model.layers.0.self_attn.k_proj.weight")
            .unwrap();
        assert_eq!(k_shape, &vec![4, 4]);
        assert_eq!(k_data[0], 16.0);

        let (v_data, v_shape) = tensors
            .get("model.layers.0.self_attn.v_proj.weight")
            .unwrap();
        assert_eq!(v_shape, &vec![4, 4]);
        assert_eq!(v_data[0], 32.0);
    }

    #[test]
    fn test_split_gpt2_fused_qkv_weight_not_2d_passthrough() {
        let mut tensors = BTreeMap::new();
        // 3D tensor: should not split
        tensors.insert(
            "model.layers.0.self_attn.c_attn.weight".to_string(),
            (vec![1.0f32; 24], vec![2, 3, 4]),
        );

        Architecture::split_gpt2_fused_qkv(&mut tensors);
        assert_eq!(tensors.len(), 1);
        assert!(tensors.contains_key("model.layers.0.self_attn.c_attn.weight"));
    }

    #[test]
    fn test_split_gpt2_fused_qkv_weight_rows_not_divisible_by_3_passthrough() {
        let mut tensors = BTreeMap::new();
        // shape[0]=5 not divisible by 3, and shape[1]!=3*shape[0]
        tensors.insert(
            "model.layers.0.self_attn.c_attn.weight".to_string(),
            (vec![1.0f32; 20], vec![5, 4]),
        );

        Architecture::split_gpt2_fused_qkv(&mut tensors);
        assert_eq!(tensors.len(), 1);
        assert!(tensors.contains_key("model.layers.0.self_attn.c_attn.weight"));
    }

    #[test]
    fn test_split_gpt2_fused_qkv_no_c_attn_tensors() {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![1.0f32; 16], vec![4, 4]),
        );
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (vec![2.0f32; 16], vec![4, 4]),
        );

        Architecture::split_gpt2_fused_qkv(&mut tensors);
        // No c_attn tensors -> no changes
        assert_eq!(tensors.len(), 2);
    }

    #[test]
    fn test_split_gpt2_fused_qkv_multiple_layers() {
        let mut tensors = BTreeMap::new();
        // Layer 0 and layer 1, both with fused c_attn bias
        for layer in 0..2 {
            let data: Vec<f32> = (0..9).map(|i| (layer * 100 + i) as f32).collect();
            tensors.insert(
                format!("model.layers.{layer}.self_attn.c_attn.bias"),
                (data, vec![9]),
            );
        }

        Architecture::split_gpt2_fused_qkv(&mut tensors);
        // 2 fused -> 6 split (3 per layer)
        assert_eq!(tensors.len(), 6);
        assert!(tensors.contains_key("model.layers.0.self_attn.q_proj.bias"));
        assert!(tensors.contains_key("model.layers.1.self_attn.v_proj.bias"));
    }

    // ====================================================================
    // split_gpt2_fused_qkv_raw: Coverage tests (impact 6.7)
    // ====================================================================

    #[test]
    fn test_split_gpt2_fused_qkv_raw_weight_split() {
        use crate::format::gguf::GgufRawTensor;

        let mut tensors = BTreeMap::new();
        // Weight: [3*hidden, hidden] = [12, 4], raw bytes
        let raw_data: Vec<u8> = (0..48).collect(); // 48 bytes
        tensors.insert(
            "model.layers.0.self_attn.c_attn.weight".to_string(),
            GgufRawTensor {
                data: raw_data,
                shape: vec![12, 4],
                dtype: 0, // F32
            },
        );

        Architecture::split_gpt2_fused_qkv_raw(&mut tensors);

        assert_eq!(tensors.len(), 3);

        let q = tensors
            .get("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(q.shape, vec![4, 4]);
        assert_eq!(q.data.len(), 16);
        assert_eq!(q.dtype, 0);

        let k = tensors
            .get("model.layers.0.self_attn.k_proj.weight")
            .unwrap();
        assert_eq!(k.shape, vec![4, 4]);
        assert_eq!(k.data.len(), 16);

        let v = tensors
            .get("model.layers.0.self_attn.v_proj.weight")
            .unwrap();
        assert_eq!(v.shape, vec![4, 4]);
        assert_eq!(v.data.len(), 16);
    }

    #[test]
    fn test_split_gpt2_fused_qkv_raw_bias_split() {
        use crate::format::gguf::GgufRawTensor;

        let mut tensors = BTreeMap::new();
        // Bias: 1D [12], raw bytes of length 24 (2 bytes per element)
        let raw_data: Vec<u8> = (0..24).collect();
        tensors.insert(
            "model.layers.0.self_attn.c_attn.bias".to_string(),
            GgufRawTensor {
                data: raw_data,
                shape: vec![12],
                dtype: 1, // F16
            },
        );

        Architecture::split_gpt2_fused_qkv_raw(&mut tensors);

        assert_eq!(tensors.len(), 3);

        let q = tensors.get("model.layers.0.self_attn.q_proj.bias").unwrap();
        assert_eq!(q.shape, vec![4]);
        assert_eq!(q.data.len(), 8);
        assert_eq!(q.dtype, 1);

        let k = tensors.get("model.layers.0.self_attn.k_proj.bias").unwrap();
        assert_eq!(k.shape, vec![4]);
        assert_eq!(k.data.len(), 8);

        let v = tensors.get("model.layers.0.self_attn.v_proj.bias").unwrap();
        assert_eq!(v.shape, vec![4]);
        assert_eq!(v.data.len(), 8);
    }

    #[test]
    fn test_split_gpt2_fused_qkv_raw_bias_not_divisible_passthrough() {
        use crate::format::gguf::GgufRawTensor;

        let mut tensors = BTreeMap::new();
        // Bias with shape not divisible by 3
        tensors.insert(
            "model.layers.0.self_attn.c_attn.bias".to_string(),
            GgufRawTensor {
                data: vec![0u8; 7],
                shape: vec![7],
                dtype: 0,
            },
        );

        Architecture::split_gpt2_fused_qkv_raw(&mut tensors);
        assert_eq!(tensors.len(), 1);
        assert!(tensors.contains_key("model.layers.0.self_attn.c_attn.bias"));
    }

    #[test]
    fn test_split_gpt2_fused_qkv_raw_weight_not_2d_passthrough() {
        use crate::format::gguf::GgufRawTensor;

        let mut tensors = BTreeMap::new();
        // 1D weight: should not split
        tensors.insert(
            "model.layers.0.self_attn.c_attn.weight".to_string(),
            GgufRawTensor {
                data: vec![0u8; 12],
                shape: vec![12],
                dtype: 0,
            },
        );

        Architecture::split_gpt2_fused_qkv_raw(&mut tensors);
        assert_eq!(tensors.len(), 1);
    }

    #[test]
    fn test_split_gpt2_fused_qkv_raw_weight_rows_not_divisible_passthrough() {
        use crate::format::gguf::GgufRawTensor;

        let mut tensors = BTreeMap::new();
        // shape[0]=5, not divisible by 3
        tensors.insert(
            "model.layers.0.self_attn.c_attn.weight".to_string(),
            GgufRawTensor {
                data: vec![0u8; 20],
                shape: vec![5, 4],
                dtype: 0,
            },
        );

        Architecture::split_gpt2_fused_qkv_raw(&mut tensors);
        assert_eq!(tensors.len(), 1);
    }
