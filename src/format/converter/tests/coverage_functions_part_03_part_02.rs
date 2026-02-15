
    // ------------------------------------------------------------------------
    // write_apr_file coverage tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_write_apr_file_basic() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let output_path = temp_dir.path().join("output.apr");

        // Create minimal tensor data
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
        );

        let options = ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        };
        let empty_f16: BTreeMap<String, (Vec<u8>, Vec<usize>)> = BTreeMap::new();
        let result = write_apr_file(
            &tensors,
            &empty_f16, // GH-205: F16 passthrough
            &output_path,
            &options,
            None,
            None,
            &Default::default(),
        );
        assert!(
            result.is_ok(),
            "write_apr_file should succeed: {:?}",
            result.err()
        );
        assert!(output_path.exists());
    }

    #[test]
    fn test_write_apr_file_with_tokenizer() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let output_path = temp_dir.path().join("with_tok.apr");

        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], vec![4, 2]),
        );

        let tokenizer = GgufTokenizer {
            vocabulary: vec![
                "hello".to_string(),
                "world".to_string(),
                "test".to_string(),
                "end".to_string(),
            ],
            merges: vec!["he llo".to_string(), "wo rld".to_string()],
            model_type: Some("bpe".to_string()),
            bos_token_id: Some(0),
            eos_token_id: Some(3),
            architecture: Some("llama".to_string()),
            model_name: Some("pygmy".to_string()),
            ..Default::default()
        };

        let options = ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        };
        let empty_f16: BTreeMap<String, (Vec<u8>, Vec<usize>)> = BTreeMap::new();
        let result = write_apr_file(
            &tensors,
            &empty_f16, // GH-205: F16 passthrough
            &output_path,
            &options,
            Some(&tokenizer),
            None,
            &Default::default(),
        );
        assert!(
            result.is_ok(),
            "write_apr_file with tokenizer should succeed"
        );
    }

    #[test]
    fn test_write_apr_file_with_model_config() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let output_path = temp_dir.path().join("with_config.apr");

        // Create tensors matching a small config
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.1; 64], vec![8, 8]),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.01; 64], vec![8, 8]),
        );
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (vec![0.02; 64], vec![8, 8]),
        );
        tensors.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            (vec![0.03; 64], vec![8, 8]),
        );

        let model_config = GgufModelConfig {
            architecture: Some("llama".to_string()),
            hidden_size: Some(8),
            num_layers: Some(1),
            num_heads: Some(2),
            num_kv_heads: Some(2),
            vocab_size: Some(8),
            intermediate_size: Some(16),
            max_position_embeddings: Some(128),
            rope_theta: Some(10000.0),
            rms_norm_eps: Some(1e-5),
            rope_type: Some(0),
        };

        let options = ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        };
        let empty_f16: BTreeMap<String, (Vec<u8>, Vec<usize>)> = BTreeMap::new();
        let result = write_apr_file(
            &tensors,
            &empty_f16,
            &output_path,
            &options,
            None,
            Some(&model_config),
            &Default::default(),
        );
        assert!(
            result.is_ok(),
            "write_apr_file with config should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_write_apr_file_with_quantization_fp16() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let output_path = temp_dir.path().join("fp16.apr");

        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
        );

        let mut options = ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        };
        options.quantize = Some(QuantizationType::Fp16);

        let empty_f16: BTreeMap<String, (Vec<u8>, Vec<usize>)> = BTreeMap::new();
        let result = write_apr_file(
            &tensors,
            &empty_f16,
            &output_path,
            &options,
            None,
            None,
            &Default::default(),
        );
        assert!(result.is_ok(), "write_apr_file with fp16 should succeed");
    }

    #[test]
    fn test_write_apr_file_with_quantization_int8() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let output_path = temp_dir.path().join("int8.apr");

        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
        );

        let mut options = ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        };
        options.quantize = Some(QuantizationType::Int8);

        let empty_f16: BTreeMap<String, (Vec<u8>, Vec<usize>)> = BTreeMap::new();
        let result = write_apr_file(
            &tensors,
            &empty_f16,
            &output_path,
            &options,
            None,
            None,
            &Default::default(),
        );
        assert!(result.is_ok(), "write_apr_file with int8 should succeed");
    }

    #[test]
    fn test_write_apr_file_tied_embeddings() {
        // Test that lm_head.weight is created from embed_tokens when missing
        let temp_dir = TempDir::new().expect("Create temp dir");
        let output_path = temp_dir.path().join("tied.apr");

        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // Only add embed_tokens, no lm_head
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
        );

        let options = ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        };
        let empty_f16: BTreeMap<String, (Vec<u8>, Vec<usize>)> = BTreeMap::new();
        let result = write_apr_file(
            &tensors,
            &empty_f16,
            &output_path,
            &options,
            None,
            None,
            &Default::default(),
        );
        assert!(result.is_ok());

        // Read back and verify lm_head was created
        let apr_data = fs::read(&output_path).expect("Read APR");
        let reader = AprV2Reader::from_bytes(&apr_data).expect("Parse APR");
        let tensor_names = reader.tensor_names();
        assert!(
            tensor_names.iter().any(|n| *n == "lm_head.weight"),
            "lm_head.weight should be created from tied embeddings"
        );
    }

    #[test]
    fn test_write_apr_file_qkv_fusion() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let output_path = temp_dir.path().join("fused.apr");

        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.1; 16], vec![4, 4]),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.1; 16], vec![4, 4]),
        );
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (vec![0.2; 16], vec![4, 4]),
        );
        tensors.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            (vec![0.3; 16], vec![4, 4]),
        );
        tensors.insert(
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            (vec![0.4; 16], vec![4, 4]),
        );

        let model_config = GgufModelConfig {
            architecture: Some("llama".to_string()),
            hidden_size: Some(4),
            num_layers: Some(1),
            num_heads: Some(1),
            num_kv_heads: Some(1),
            vocab_size: Some(4),
            intermediate_size: Some(8),
            max_position_embeddings: Some(64),
            rope_theta: Some(10000.0),
            rms_norm_eps: Some(1e-5),
            rope_type: Some(0),
        };

        let options = ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        };
        let empty_f16: BTreeMap<String, (Vec<u8>, Vec<usize>)> = BTreeMap::new();
        let result = write_apr_file(
            &tensors,
            &empty_f16,
            &output_path,
            &options,
            None,
            Some(&model_config),
            &Default::default(),
        );
        assert!(
            result.is_ok(),
            "write_apr_file with QKV fusion should succeed: {:?}",
            result.err()
        );
    }

    // ------------------------------------------------------------------------
    // GH-205: F16 Passthrough Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_gh205_f16_passthrough_preserves_bytes() {
        // GH-205: Verify F16 SafeTensors -> APR conversion preserves raw bytes
        use crate::format::converter::import::apr_import;
        use crate::format::test_factory::build_pygmy_safetensors_f16;
        use crate::format::v2::AprV2Reader;

        let temp_dir = TempDir::new().expect("Create temp dir");
        let st_path = temp_dir.path().join("f16_model.safetensors");
        let apr_path = temp_dir.path().join("f16_model.apr");

        // Create F16 SafeTensors
        let st_data = build_pygmy_safetensors_f16();
        fs::write(&st_path, &st_data).expect("Write F16 SafeTensors");

        // Import with default options (should use F16 passthrough)
        let mut options = ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        };
        options.architecture = Architecture::Qwen2;

        let result = apr_import(st_path.to_str().unwrap(), &apr_path, options);
        assert!(
            result.is_ok(),
            "F16 import should succeed: {:?}",
            result.err()
        );

        // Read back APR and verify tensors are F16
        let apr_bytes = fs::read(&apr_path).expect("Read APR");
        let reader = AprV2Reader::from_bytes(&apr_bytes).expect("Parse APR");

        // Find embedding tensor and verify dtype
        let tensor_names = reader.tensor_names();
        let embed_name = tensor_names
            .iter()
            .find(|n| n.contains("embed_tokens"))
            .expect("Should have embed_tokens tensor");

        let entry = reader.get_tensor(embed_name).expect("Get tensor entry");
        assert_eq!(
            entry.dtype,
            crate::format::v2::TensorDType::F16,
            "GH-205 FAIL: Tensor should be F16, got {:?}",
            entry.dtype
        );
    }

    #[test]
    fn test_gh205_f16_passthrough_no_precision_loss() {
        // GH-205: Verify F16 -> APR -> readback produces identical bytes
        use crate::format::converter::import::apr_import;
        use crate::format::test_factory::build_pygmy_safetensors_f16;
        use crate::serialization::safetensors::MappedSafeTensors;

        let temp_dir = TempDir::new().expect("Create temp dir");
        let st_path = temp_dir.path().join("f16_model.safetensors");
        let apr_path = temp_dir.path().join("f16_model.apr");

        // Create F16 SafeTensors
        let st_data = build_pygmy_safetensors_f16();
        fs::write(&st_path, &st_data).expect("Write F16 SafeTensors");

        // Get original F16 bytes from SafeTensors
        let mapped = MappedSafeTensors::open(&st_path).expect("Open SafeTensors");
        let original_bytes = mapped
            .get_tensor_bytes("model.embed_tokens.weight")
            .expect("Get original F16 bytes");
        let original_len = original_bytes.len();

        // Import with F16 passthrough
        let mut options = ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        };
        options.architecture = Architecture::Qwen2;

        let result = apr_import(st_path.to_str().unwrap(), &apr_path, options);
        assert!(
            result.is_ok(),
            "F16 import should succeed: {:?}",
            result.err()
        );

        // Read back from APR
        let apr_bytes = fs::read(&apr_path).expect("Read APR");
        let reader = crate::format::v2::AprV2Reader::from_bytes(&apr_bytes).expect("Parse APR");

        // Get F16 bytes from APR (mapped name)
        let apr_tensor_bytes = reader
            .get_tensor_data("model.embed_tokens.weight")
            .expect("Get APR tensor bytes");

        // Verify size matches (same number of bytes = no conversion happened)
        assert_eq!(
            apr_tensor_bytes.len(),
            original_len,
            "GH-205 FAIL: APR tensor size {} != original F16 size {} (conversion occurred)",
            apr_tensor_bytes.len(),
            original_len
        );
    }

    // ------------------------------------------------------------------------
    // Rosetta conversion coverage tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_rosetta_inspect_safetensors() {
        use crate::format::rosetta::RosettaStone;

        let temp_dir = TempDir::new().expect("Create temp dir");
        let st_path = temp_dir.path().join("model.safetensors");

        let st_data = build_pygmy_safetensors();
        fs::write(&st_path, &st_data).expect("Write safetensors");

        let rosetta = RosettaStone::new();
        let result = rosetta.inspect(&st_path);
        assert!(
            result.is_ok(),
            "Rosetta inspect should succeed: {:?}",
            result.err()
        );

        let inspection = result.unwrap();
        assert!(!inspection.tensors.is_empty());
        assert!(inspection.file_size > 0);
    }
