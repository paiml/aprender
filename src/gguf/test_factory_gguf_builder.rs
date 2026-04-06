
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GGUFModel;

    #[test]
    fn test_gguf_builder_empty() {
        let data = GGUFBuilder::new().build();

        // Should have valid header
        assert!(data.len() >= 24); // magic + version + 2 counts

        let model = GGUFModel::from_bytes(&data).expect("Should parse empty GGUF");
        assert_eq!(model.header.magic, GGUF_MAGIC);
        assert_eq!(model.header.version, GGUF_VERSION_V3);
        assert_eq!(model.metadata.len(), 0);
        assert_eq!(model.tensors.len(), 0);
    }

    #[test]
    fn test_gguf_builder_metadata_only() {
        let data = GGUFBuilder::new()
            .architecture("llama")
            .add_u32("test.value", 42)
            .add_f32("test.float", 3.14)
            .build();

        let model = GGUFModel::from_bytes(&data).expect("Should parse");
        assert_eq!(model.metadata.len(), 3);
        assert_eq!(model.architecture(), Some("llama"));
    }

    #[test]
    fn test_gguf_builder_with_tensor() {
        let data = GGUFBuilder::new()
            .add_f32_tensor("test.weight", &[4, 8], &vec![0.0f32; 32])
            .build();

        let model = GGUFModel::from_bytes(&data).expect("Should parse");
        assert_eq!(model.tensors.len(), 1);
        assert_eq!(model.tensors[0].name, "test.weight");
        assert_eq!(model.tensors[0].n_dims, 2);
    }

    #[test]
    fn test_gguf_builder_q4_k_tensor() {
        let q4k_data = create_q4_k_data(256);
        let data = GGUFBuilder::new()
            .add_q4_k_tensor("layer.weight", &[256], &q4k_data)
            .build();

        let model = GGUFModel::from_bytes(&data).expect("Should parse");
        assert_eq!(model.tensors[0].qtype, GGUF_TYPE_Q4_K);
    }

    #[test]
    fn test_minimal_llama_model() {
        let data = build_minimal_llama_gguf(100, 64, 128, 4, 4);

        let model = GGUFModel::from_bytes(&data).expect("Should parse minimal LLaMA");

        assert_eq!(model.architecture(), Some("llama"));
        assert_eq!(model.embedding_dim(), Some(64));
        assert_eq!(model.num_layers(), Some(1));
        assert_eq!(model.num_heads(), Some(4));

        // Should have all expected tensors
        let tensor_names: Vec<_> = model.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(tensor_names.contains(&"token_embd.weight"));
        assert!(tensor_names.contains(&"blk.0.attn_q.weight"));
        assert!(tensor_names.contains(&"blk.0.ffn_up.weight"));
        assert!(tensor_names.contains(&"output_norm.weight"));
    }

    #[test]
    fn test_gguf_builder_default() {
        let builder = GGUFBuilder::default();
        let data = builder.build();
        let model = GGUFModel::from_bytes(&data).expect("Should parse default builder");
        assert_eq!(model.tensors.len(), 0);
    }

    #[test]
    fn test_create_q4_0_data_size() {
        // 32 elements -> 1 block -> 18 bytes
        let data = create_q4_0_data(32);
        assert_eq!(data.len(), 18);

        // 64 elements -> 2 blocks -> 36 bytes
        let data = create_q4_0_data(64);
        assert_eq!(data.len(), 36);

        // 33 elements -> 2 blocks (ceil) -> 36 bytes
        let data = create_q4_0_data(33);
        assert_eq!(data.len(), 36);
    }

    #[test]
    fn test_create_q8_0_data_size() {
        // 32 elements -> 1 block -> 34 bytes
        let data = create_q8_0_data(32);
        assert_eq!(data.len(), 34);

        // 64 elements -> 2 blocks -> 68 bytes
        let data = create_q8_0_data(64);
        assert_eq!(data.len(), 68);
    }

    #[test]
    fn test_create_q4_k_data_size() {
        // 256 elements -> 1 super-block -> 144 bytes
        let data = create_q4_k_data(256);
        assert_eq!(data.len(), 144);

        // 512 elements -> 2 super-blocks -> 288 bytes
        let data = create_q4_k_data(512);
        assert_eq!(data.len(), 288);

        // 257 elements -> 2 super-blocks (ceil) -> 288 bytes
        let data = create_q4_k_data(257);
        assert_eq!(data.len(), 288);
    }

    #[test]
    fn test_create_q5_k_data_size() {
        // 256 elements -> 1 super-block -> 176 bytes
        let data = create_q5_k_data(256);
        assert_eq!(data.len(), 176);

        // 512 elements -> 2 super-blocks -> 352 bytes
        let data = create_q5_k_data(512);
        assert_eq!(data.len(), 352);
    }

    #[test]
    fn test_create_q6_k_data_size() {
        // 256 elements -> 1 super-block -> 210 bytes
        let data = create_q6_k_data(256);
        assert_eq!(data.len(), 210);

        // 512 elements -> 2 super-blocks -> 420 bytes
        let data = create_q6_k_data(512);
        assert_eq!(data.len(), 420);
    }

    #[test]
    fn test_create_f32_embedding_data() {
        let data = create_f32_embedding_data(10, 8);
        assert_eq!(data.len(), 80);
        // Values should be deterministic
        let first = data[0];
        let second = data[1];
        assert!((first - (-500.0 / 5000.0)).abs() < 1e-6);
        assert!((second - (-499.0 / 5000.0)).abs() < 1e-6);
    }

    #[test]
    fn test_create_f32_norm_weights() {
        let data = create_f32_norm_weights(64);
        assert_eq!(data.len(), 64);
        assert!(data.iter().all(|&v| (v - 1.0).abs() < f32::EPSILON));
    }

    #[test]
    fn test_gguf_builder_q4_0_tensor() {
        let q4_data = create_q4_0_data(64);
        let data = GGUFBuilder::new()
            .add_q4_0_tensor("test.q4_0", &[64], &q4_data)
            .build();

        let model = GGUFModel::from_bytes(&data).expect("Should parse");
        assert_eq!(model.tensors[0].qtype, GGUF_TYPE_Q4_0);
    }

    #[test]
    fn test_gguf_builder_q8_0_tensor() {
        let q8_data = create_q8_0_data(64);
        let data = GGUFBuilder::new()
            .add_q8_0_tensor("test.q8_0", &[64], &q8_data)
            .build();

        let model = GGUFModel::from_bytes(&data).expect("Should parse");
        assert_eq!(model.tensors[0].qtype, GGUF_TYPE_Q8_0);
    }

    #[test]
    fn test_gguf_builder_q5_k_tensor() {
        let q5k_data = create_q5_k_data(256);
        let data = GGUFBuilder::new()
            .add_q5_k_tensor("test.q5_k", &[256], &q5k_data)
            .build();

        let model = GGUFModel::from_bytes(&data).expect("Should parse");
        assert_eq!(model.tensors[0].qtype, GGUF_TYPE_Q5_K);
    }

    #[test]
    fn test_gguf_builder_q6_k_tensor() {
        let q6k_data = create_q6_k_data(256);
        let data = GGUFBuilder::new()
            .add_q6_k_tensor("test.q6_k", &[256], &q6k_data)
            .build();

        let model = GGUFModel::from_bytes(&data).expect("Should parse");
        assert_eq!(model.tensors[0].qtype, GGUF_TYPE_Q6_K);
    }

    #[test]
    fn test_gguf_builder_multiple_tensors() {
        let data = GGUFBuilder::new()
            .add_f32_tensor("a", &[4], &[1.0, 2.0, 3.0, 4.0])
            .add_f32_tensor("b", &[2, 2], &[1.0, 2.0, 3.0, 4.0])
            .add_f32_tensor("c", &[1], &[42.0])
            .build();

        let model = GGUFModel::from_bytes(&data).expect("Should parse");
        assert_eq!(model.tensors.len(), 3);
    }

    #[test]
    fn test_minimal_phi2_model() {
        let data = build_minimal_phi2_gguf(100, 64, 128, 4);

        let model = GGUFModel::from_bytes(&data).expect("Should parse minimal Phi-2");

        assert_eq!(model.architecture(), Some("phi2"));
        assert_eq!(model.embedding_dim(), Some(64));
        assert_eq!(model.num_layers(), Some(1));
        assert_eq!(model.num_heads(), Some(4));

        // Phi-2 has fused QKV
        let tensor_names: Vec<_> = model.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(tensor_names.contains(&"blk.0.attn_qkv.weight"));
        // No separate Q/K/V weights
        assert!(!tensor_names.contains(&"blk.0.attn_q.weight"));
    }

    #[test]
    fn test_gguf_builder_all_metadata_setters() {
        let data = GGUFBuilder::new()
            .architecture("test_arch")
            .hidden_dim("test_arch", 512)
            .num_layers("test_arch", 12)
            .num_heads("test_arch", 8)
            .num_kv_heads("test_arch", 2)
            .context_length("test_arch", 4096)
            .rope_freq_base("test_arch", 10000.0)
            .rms_epsilon("test_arch", 1e-5)
            .ffn_hidden_dim("test_arch", 2048)
            .vocab_size("test_arch", 32000)
            .build();

        let model = GGUFModel::from_bytes(&data).expect("Should parse");
        assert_eq!(model.architecture(), Some("test_arch"));
        assert_eq!(model.embedding_dim(), Some(512));
        assert_eq!(model.num_layers(), Some(12));
    }

    #[test]
    fn test_minimal_llama_with_gqa() {
        // LLaMA 2 style with GQA (8 KV heads vs 32 Q heads)
        let data = build_minimal_llama_gguf(100, 128, 256, 8, 2);

        let model = GGUFModel::from_bytes(&data).expect("Should parse");
        assert_eq!(model.num_heads(), Some(8));
        assert_eq!(model.num_kv_heads(), Some(2));
    }

    #[test]
    fn test_create_quantized_data_small() {
        // Test with very small sizes
        let q4_0 = create_q4_0_data(1);
        assert_eq!(q4_0.len(), 18); // At least 1 block

        let q8_0 = create_q8_0_data(1);
        assert_eq!(q8_0.len(), 34); // At least 1 block

        let q4_k = create_q4_k_data(1);
        assert_eq!(q4_k.len(), 144); // At least 1 super-block
    }

    #[test]
    fn test_gguf_builder_chaining() {
        // Test method chaining works correctly
        let builder = GGUFBuilder::new()
            .architecture("test")
            .add_u32("a", 1)
            .add_u32("b", 2)
            .add_f32("c", 3.0)
            .add_string("d", "value");

        let data = builder.build();
        let model = GGUFModel::from_bytes(&data).expect("Should parse");
        assert_eq!(model.metadata.len(), 5); // arch + 4 added
    }

    // =========================================================================
    // T-COV-95: Active Pygmy (Executable) Tests
    // =========================================================================

    #[test]
    fn test_executable_pygmy_parses() {
        // T-COV-95: Verify the executable pygmy GGUF is valid
        let data = build_executable_pygmy_gguf();
        let model = GGUFModel::from_bytes(&data).expect("Executable pygmy should parse");

        assert_eq!(model.architecture(), Some("llama"));
        assert_eq!(model.embedding_dim(), Some(32));
        assert_eq!(model.num_layers(), Some(1));
        assert_eq!(model.num_heads(), Some(4));
        assert_eq!(model.num_kv_heads(), Some(4));

        // Vocab size is 32 (from token_embd.weight shape)
        let token_embd = model.tensors.iter().find(|t| t.name == "token_embd.weight");
        assert!(token_embd.is_some());
        assert_eq!(token_embd.unwrap().dims[0], 32); // vocab_size
    }

    #[test]
    fn test_executable_pygmy_has_all_tensors() {
        // T-COV-95: Verify all required tensors are present
        let data = build_executable_pygmy_gguf();
        let model = GGUFModel::from_bytes(&data).expect("Should parse");

        let tensor_names: Vec<_> = model.tensors.iter().map(|t| t.name.as_str()).collect();

        // Required tensors for forward()
        assert!(
            tensor_names.contains(&"token_embd.weight"),
            "Missing token_embd"
        );
        assert!(
            tensor_names.contains(&"blk.0.attn_norm.weight"),
            "Missing attn_norm"
        );
        assert!(
            tensor_names.contains(&"blk.0.attn_q.weight"),
            "Missing attn_q"
        );
        assert!(
            tensor_names.contains(&"blk.0.attn_k.weight"),
            "Missing attn_k"
        );
        assert!(
            tensor_names.contains(&"blk.0.attn_v.weight"),
            "Missing attn_v"
        );
        assert!(
            tensor_names.contains(&"blk.0.attn_output.weight"),
            "Missing attn_output"
        );
        assert!(
            tensor_names.contains(&"blk.0.ffn_norm.weight"),
            "Missing ffn_norm"
        );
        assert!(
            tensor_names.contains(&"blk.0.ffn_gate.weight"),
            "Missing ffn_gate"
        );
        assert!(
            tensor_names.contains(&"blk.0.ffn_up.weight"),
            "Missing ffn_up"
        );
        assert!(
            tensor_names.contains(&"blk.0.ffn_down.weight"),
            "Missing ffn_down"
        );
        assert!(
            tensor_names.contains(&"output_norm.weight"),
            "Missing output_norm"
        );
    }

    #[test]
    fn test_executable_pygmy_tensor_dimensions() {
        // T-COV-95: Verify tensor dimensions are correct
        let data = build_executable_pygmy_gguf();
        let model = GGUFModel::from_bytes(&data).expect("Should parse");

        for tensor in &model.tensors {
            match tensor.name.as_str() {
                "token_embd.weight" => {
                    // [vocab_size=32, hidden_dim=32]
                    assert_eq!(tensor.dims.len(), 2);
                    assert_eq!(tensor.dims[0], 32);
                    assert_eq!(tensor.dims[1], 32);
                },
                "blk.0.attn_q.weight" | "blk.0.attn_output.weight" => {
                    // [hidden_dim=32, hidden_dim=32] - Q4_0 quantized
                    assert_eq!(tensor.dims.len(), 2);
                    assert_eq!(tensor.dims[0], 32);
                    assert_eq!(tensor.dims[1], 32);
                },
                "blk.0.ffn_gate.weight" | "blk.0.ffn_up.weight" => {
                    // [hidden_dim=32, intermediate_dim=64] - Q4_0 quantized
                    assert_eq!(tensor.dims.len(), 2);
                    assert_eq!(tensor.dims[0], 32);
                    assert_eq!(tensor.dims[1], 64);
                },
                "blk.0.ffn_down.weight" => {
                    // [intermediate_dim=64, hidden_dim=32] - Q4_0 quantized
                    assert_eq!(tensor.dims.len(), 2);
                    assert_eq!(tensor.dims[0], 64);
                    assert_eq!(tensor.dims[1], 32);
                },
                _ => {}, // Other tensors checked elsewhere
            }
        }
    }

    #[test]
    fn test_executable_pygmy_tensor_types() {
        // T-COV-95: Verify correct quantization types per tensor
        let data = build_executable_pygmy_gguf();
        let model = GGUFModel::from_bytes(&data).expect("Should parse");

        for tensor in &model.tensors {
            match tensor.name.as_str() {
                // F32 tensors: embeddings and norms
                "token_embd.weight"
                | "blk.0.attn_norm.weight"
                | "blk.0.ffn_norm.weight"
                | "output_norm.weight" => {
                    assert_eq!(
                        tensor.qtype, GGUF_TYPE_F32,
                        "Tensor {} should be F32, got qtype {}",
                        tensor.name, tensor.qtype
                    );
                },
                // Q4_0 tensors: weight matrices
                _ => {
                    assert_eq!(
                        tensor.qtype, GGUF_TYPE_Q4_0,
                        "Tensor {} should be Q4_0, got qtype {}",
                        tensor.name, tensor.qtype
                    );
                },
            }
        }
    }

    #[test]
    fn test_executable_pygmy_size() {
        // T-COV-95: Verify model is small (the "Pygmy" property)
        let data = build_executable_pygmy_gguf();

        // Should be under 50KB (actual ~15-20KB with F32 tensors)
        assert!(
            data.len() < 50_000,
            "Executable pygmy should be < 50KB, got {} bytes",
            data.len()
        );

        // But should be non-trivial (contains actual weights)
        assert!(
            data.len() > 1_000,
            "Executable pygmy should be > 1KB, got {} bytes",
            data.len()
        );
    }
}
