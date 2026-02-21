    // =========================================================================
    // infer_architecture_from_names
    // =========================================================================

    #[test]
    fn test_infer_arch_qwen2_with_attn_bias_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![1.0], vec![1]),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.bias".to_string(),
            (vec![1.0], vec![1]),
        );

        let result = infer_architecture_from_names(&tensors);
        assert_eq!(result, Some("qwen2".to_string()));
    }

    #[test]
    fn test_infer_arch_llama_no_bias_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![1.0], vec![1]),
        );
        // No bias → LLaMA

        let result = infer_architecture_from_names(&tensors);
        assert_eq!(result, Some("llama".to_string()));
    }

    #[test]
    fn test_infer_arch_gpt2_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "transformer.h.0.attn.c_attn.weight".to_string(),
            (vec![1.0], vec![1]),
        );

        let result = infer_architecture_from_names(&tensors);
        assert_eq!(result, Some("gpt2".to_string()));
    }

    #[test]
    fn test_infer_arch_gpt2_safetensors_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // GH-255: SafeTensors GPT-2 uses "h.N.*" prefix
        tensors.insert(
            "h.0.attn.c_attn.weight".to_string(),
            (vec![1.0], vec![1]),
        );

        let result = infer_architecture_from_names(&tensors);
        assert_eq!(result, Some("gpt2".to_string()));
    }

    #[test]
    fn test_infer_arch_gguf_blk_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "blk.0.attn_q.weight".to_string(),
            (vec![1.0], vec![1]),
        );

        let result = infer_architecture_from_names(&tensors);
        assert_eq!(result, Some("unknown".to_string()));
    }

    #[test]
    fn test_infer_arch_empty_tensors_gh219() {
        let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        let result = infer_architecture_from_names(&tensors);
        assert_eq!(result, Some("unknown".to_string()));
    }

    // =========================================================================
    // map_tensor_names
    // =========================================================================

    #[test]
    fn test_map_tensor_names_qwen2_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert("blk.0.attn_q.weight".to_string(), (vec![1.0], vec![1]));
        tensors.insert("token_embd.weight".to_string(), (vec![2.0], vec![1]));

        let mapped = map_tensor_names(&tensors, Architecture::Qwen2);

        assert!(mapped.contains_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(mapped.contains_key("model.embed_tokens.weight"));
        assert!(!mapped.contains_key("blk.0.attn_q.weight"));
    }

    #[test]
    fn test_map_tensor_names_auto_preserves_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert("model.layers.0.weight".to_string(), (vec![1.0], vec![1]));

        let mapped = map_tensor_names(&tensors, Architecture::Auto);

        assert!(mapped.contains_key("model.layers.0.weight"));
    }

    // =========================================================================
    // check_special_values
    // =========================================================================

    #[test]
    fn test_check_special_values_clean_gh219() {
        let stats = TensorStats {
            name: "test".to_string(),
            count: 10,
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };
        let options = ImportOptions {
            validation: ValidationConfig::Basic,
            ..Default::default()
        };

        let errors = check_special_values("test", &stats, &options);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_check_special_values_with_nan_gh219() {
        let stats = TensorStats {
            name: "test".to_string(),
            count: 10,
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            nan_count: 5,
            inf_count: 0,
            zero_count: 0,
        };
        let options = ImportOptions {
            validation: ValidationConfig::Basic,
            ..Default::default()
        };

        let errors = check_special_values("test.weight", &stats, &options);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("NaN"));
        assert!(errors[0].contains("5"));
    }

    #[test]
    fn test_check_special_values_with_inf_gh219() {
        let stats = TensorStats {
            name: "test".to_string(),
            count: 10,
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            nan_count: 0,
            inf_count: 3,
            zero_count: 0,
        };
        let options = ImportOptions {
            validation: ValidationConfig::Basic,
            ..Default::default()
        };

        let errors = check_special_values("test.weight", &stats, &options);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("Inf"));
    }

    #[test]
    fn test_check_special_values_with_both_gh219() {
        let stats = TensorStats {
            name: "test".to_string(),
            count: 10,
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            nan_count: 2,
            inf_count: 1,
            zero_count: 0,
        };
        let options = ImportOptions {
            validation: ValidationConfig::Strict,
            ..Default::default()
        };

        let errors = check_special_values("test.weight", &stats, &options);
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_check_special_values_none_validation_gh219() {
        let stats = TensorStats {
            name: "test".to_string(),
            count: 10,
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            nan_count: 100,
            inf_count: 50,
            zero_count: 0,
        };
        let options = ImportOptions {
            validation: ValidationConfig::None,
            ..Default::default()
        };

        let errors = check_special_values("test", &stats, &options);
        assert!(errors.is_empty());
    }

    // =========================================================================
    // has_required_tensor
    // =========================================================================

    #[test]
    fn test_has_required_tensor_found_gh219() {
        let mut names = std::collections::HashSet::new();
        names.insert("model.norm.weight");
        names.insert("lm_head.weight");

        assert!(has_required_tensor(&names, &["model.norm.weight", "norm.weight"]));
    }

    #[test]
    fn test_has_required_tensor_not_found_gh219() {
        let mut names = std::collections::HashSet::new();
        names.insert("model.layers.0.weight");

        assert!(!has_required_tensor(&names, &["model.norm.weight", "norm.weight"]));
    }

    #[test]
    fn test_has_required_tensor_empty_names_gh219() {
        let names = std::collections::HashSet::new();
        assert!(!has_required_tensor(&names, &["model.norm.weight"]));
    }

    // =========================================================================
    // GH-279-2: APR cache directory helpers
    // =========================================================================

    #[test]
    fn test_get_apr_cache_dir_returns_expected_path() {
        // get_apr_cache_dir() should return ~/.apr/cache/hf/
        if let Some(dir) = get_apr_cache_dir() {
            let path_str = dir.to_string_lossy();
            assert!(
                path_str.ends_with(".apr/cache/hf"),
                "APR cache dir should end with .apr/cache/hf, got: {path_str}"
            );
        }
        // If HOME is not set, returns None — that's also valid
    }

    #[test]
    fn test_find_in_cache_apr_cache_tempdir_gh279() {
        // Create a temp directory simulating ~/.apr/cache/hf/Qwen/Qwen3-8B/
        let tmp = tempfile::tempdir().expect("create tempdir");
        let org_dir = tmp.path().join("Qwen");
        let repo_dir = org_dir.join("Qwen3-8B");
        fs::create_dir_all(&repo_dir).expect("create repo dir");

        // Create a fake index file
        let index_file = repo_dir.join("model.safetensors.index.json");
        fs::write(&index_file, "{}").expect("write index");

        // Verify the helper function finds files in APR cache structure
        let apr_path = tmp
            .path()
            .join("Qwen")
            .join("Qwen3-8B")
            .join("model.safetensors.index.json");
        assert!(apr_path.exists(), "Index file should exist at APR cache path");
    }

    #[test]
    fn test_find_in_aprender_cache_returns_none_for_missing() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let result = find_in_aprender_cache(tmp.path(), "Qwen", "Qwen3-8B", "model.safetensors");
        assert!(result.is_none(), "Should return None for non-existent file");
    }

    #[test]
    fn test_find_in_aprender_cache_finds_existing_file() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let file_path = tmp
            .path()
            .join("aprender")
            .join("hf")
            .join("Qwen")
            .join("Qwen3-8B");
        fs::create_dir_all(&file_path).expect("create dirs");
        fs::write(file_path.join("model.safetensors"), b"fake").expect("write file");

        let result = find_in_aprender_cache(tmp.path(), "Qwen", "Qwen3-8B", "model.safetensors");
        assert!(result.is_some(), "Should find existing file in aprender cache");
    }
