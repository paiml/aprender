    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_finetune_method_parse() {
        assert!(matches!(
            "auto".parse::<FinetuneMethod>(),
            Ok(FinetuneMethod::Auto)
        ));
        assert!(matches!(
            "full".parse::<FinetuneMethod>(),
            Ok(FinetuneMethod::Full)
        ));
        assert!(matches!(
            "lora".parse::<FinetuneMethod>(),
            Ok(FinetuneMethod::LoRA)
        ));
        assert!(matches!(
            "qlora".parse::<FinetuneMethod>(),
            Ok(FinetuneMethod::QLoRA)
        ));
        assert!("unknown".parse::<FinetuneMethod>().is_err());
    }

    #[test]
    fn test_finetune_method_to_entrenar() {
        assert!(matches!(Method::from(FinetuneMethod::Auto), Method::Auto));
        assert!(matches!(Method::from(FinetuneMethod::LoRA), Method::LoRA));
        assert!(matches!(Method::from(FinetuneMethod::QLoRA), Method::QLoRA));
        assert!(matches!(Method::from(FinetuneMethod::Full), Method::Full));
    }

    #[test]
    fn test_parse_model_size() {
        assert_eq!(parse_model_size("7B").expect("7B"), 7_000_000_000);
        assert_eq!(parse_model_size("1.5B").expect("1.5B"), 1_500_000_000);
        assert_eq!(parse_model_size("135M").expect("135M"), 135_000_000);
        assert!(parse_model_size("invalid").is_err());
    }

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(7_000_000_000), "7.0B");
        assert_eq!(format_params(135_000_000), "135.0M");
        assert_eq!(format_params(1000), "1000");
    }

    #[test]
    fn test_run_no_model() {
        let result = run(
            None, "auto", None, 16.0, false, None, None, None, false, 3, 2e-4, None, false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_plan_with_model_size() {
        let result = run(
            None,
            "lora",
            None,
            16.0,
            true,
            None,
            None,
            None,
            false,
            3,
            2e-4,
            Some("7B"),
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_plan_json() {
        let result = run(
            None,
            "qlora",
            None,
            24.0,
            true,
            None,
            None,
            None,
            false,
            3,
            2e-4,
            Some("14B"),
            true,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_with_model_file() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 4096]).expect("write");
        let result = run(
            Some(input.path()),
            "auto",
            None,
            16.0,
            true,
            None,
            None,
            None,
            false,
            3,
            2e-4,
            None,
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_merge_no_model() {
        let result = run_merge(None, None, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_no_adapter() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run_merge(Some(input.path()), None, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_model_not_found() {
        let result = run_merge(
            Some(Path::new("/nonexistent.apr")),
            Some(Path::new("/nonexistent_adapter/")),
            None,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_is_lora_eligible() {
        assert!(is_lora_eligible("model.layers.0.self_attn.q_proj.weight"));
        assert!(is_lora_eligible("model.layers.0.self_attn.v_proj.weight"));
        assert!(is_lora_eligible("model.layers.0.mlp.gate_proj.weight"));
        assert!(is_lora_eligible("model.layers.0.mlp.up_proj.weight"));
        assert!(is_lora_eligible("model.layers.0.mlp.down_proj.weight"));
        assert!(is_lora_eligible("blk.0.attn_q.weight"));
        assert!(is_lora_eligible("blk.0.ffn_gate.weight"));

        // Should NOT be eligible
        assert!(!is_lora_eligible("model.embed_tokens.weight"));
        assert!(!is_lora_eligible("model.norm.weight"));
        assert!(!is_lora_eligible("lm_head.weight"));
        assert!(!is_lora_eligible("model.layers.0.self_attn.q_proj.bias"));
        assert!(!is_lora_eligible("token_embd.weight"));
    }

    #[test]
    fn test_hash_seed_deterministic() {
        let s1 = hash_seed("test.weight", 0);
        let s2 = hash_seed("test.weight", 0);
        assert_eq!(s1, s2, "Same inputs must produce same output");

        let s3 = hash_seed("test.weight", 1);
        assert_ne!(s1, s3, "Different index must produce different output");

        let s4 = hash_seed("other.weight", 0);
        assert_ne!(s1, s4, "Different name must produce different output");
    }

    #[test]
    fn test_run_training_creates_adapter() {
        // Create a valid model APR with LoRA-eligible layers
        let mut writer = aprender::serialization::apr::AprWriter::new();
        writer.set_metadata("model_type", serde_json::json!("test"));
        let q_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        writer.add_tensor_f32(
            "model.layers.0.self_attn.q_proj.weight",
            vec![8, 8],
            &q_data,
        );
        let v_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02).collect();
        writer.add_tensor_f32(
            "model.layers.0.self_attn.v_proj.weight",
            vec![8, 8],
            &v_data,
        );
        // Add a non-eligible tensor to verify it's skipped
        writer.add_tensor_f32("model.embed_tokens.weight", vec![10, 8], &vec![0.1; 80]);

        let input_file = NamedTempFile::with_suffix(".apr").expect("create input");
        let bytes = writer.to_bytes().expect("serialize");
        std::fs::write(input_file.path(), bytes).expect("write");

        // Create a dummy data file
        let data_file = NamedTempFile::with_suffix(".jsonl").expect("create data");
        std::fs::write(data_file.path(), "{\"text\": \"hello world\"}\n").expect("write data");

        let output_file = NamedTempFile::with_suffix(".apr").expect("create output");

        let result = run(
            Some(input_file.path()),
            "lora",
            None,
            16.0,
            false,
            Some(data_file.path()),
            Some(output_file.path()),
            None,
            false,
            3,
            2e-4,
            None,
            true,
        );
        assert!(result.is_ok(), "Training should succeed: {result:?}");

        // Verify adapter file was created and is valid APR
        let adapter = aprender::serialization::apr::AprReader::open(output_file.path())
            .expect("adapter should be valid APR");
        assert!(!adapter.tensors.is_empty(), "Adapter should have tensors");

        // Should have lora_a and lora_b for each eligible layer
        let names: Vec<&str> = adapter.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"model.layers.0.self_attn.q_proj.weight.lora_a"));
        assert!(names.contains(&"model.layers.0.self_attn.q_proj.weight.lora_b"));
        assert!(names.contains(&"model.layers.0.self_attn.v_proj.weight.lora_a"));
        assert!(names.contains(&"model.layers.0.self_attn.v_proj.weight.lora_b"));

        // Should have adapter metadata
        assert!(adapter.get_metadata("adapter_type").is_some());
        assert!(adapter.get_metadata("lora_rank").is_some());
    }

    #[test]
    fn test_merge_creates_merged_model() {
        // Create base model
        let mut base_writer = aprender::serialization::apr::AprWriter::new();
        base_writer.set_metadata("model_type", serde_json::json!("test"));
        let q_data: Vec<f32> = vec![1.0; 64];
        base_writer.add_tensor_f32(
            "model.layers.0.self_attn.q_proj.weight",
            vec![8, 8],
            &q_data,
        );
        base_writer.add_tensor_f32("model.norm.weight", vec![8], &vec![1.0; 8]);

        let base_file = NamedTempFile::with_suffix(".apr").expect("create base");
        std::fs::write(base_file.path(), base_writer.to_bytes().expect("serialize"))
            .expect("write");

        // Create adapter
        let mut adapter_writer = aprender::serialization::apr::AprWriter::new();
        adapter_writer.set_metadata("lora_rank", serde_json::json!(4));
        adapter_writer.set_metadata("lora_alpha", serde_json::json!(8.0));
        let lora_a: Vec<f32> = vec![0.1; 4 * 8]; // [rank=4, cols=8]
        adapter_writer.add_tensor_f32(
            "model.layers.0.self_attn.q_proj.weight.lora_a",
            vec![4, 8],
            &lora_a,
        );
        let lora_b: Vec<f32> = vec![0.05; 8 * 4]; // [rows=8, rank=4]
        adapter_writer.add_tensor_f32(
            "model.layers.0.self_attn.q_proj.weight.lora_b",
            vec![8, 4],
            &lora_b,
        );

        let adapter_file = NamedTempFile::with_suffix(".apr").expect("create adapter");
        std::fs::write(
            adapter_file.path(),
            adapter_writer.to_bytes().expect("serialize"),
        )
        .expect("write");

        let output_file = NamedTempFile::with_suffix(".apr").expect("create output");

        let result = run_merge(
            Some(base_file.path()),
            Some(adapter_file.path()),
            Some(output_file.path()),
            true,
        );
        assert!(result.is_ok(), "Merge should succeed: {result:?}");

        // Verify merged model
        let merged = aprender::serialization::apr::AprReader::open(output_file.path())
            .expect("merged should be valid APR");
        assert_eq!(merged.tensors.len(), 2); // q_proj + norm
        let q_merged = merged
            .read_tensor_f32("model.layers.0.self_attn.q_proj.weight")
            .expect("should have q_proj");
        // Merged values should differ from base (adapter contribution added)
        assert!(
            q_merged.iter().any(|&v| (v - 1.0).abs() > 1e-6),
            "Merged weights should differ from base"
        );
    }
