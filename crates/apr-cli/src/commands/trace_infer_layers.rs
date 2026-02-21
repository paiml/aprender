
    // ========================================================================
    // infer_layers_from_tensor_names: comprehensive
    // ========================================================================

    #[test]
    fn test_infer_layers_empty() {
        let layers = infer_layers_from_tensor_names(&[], None);
        assert!(layers.is_empty());
    }

    #[test]
    fn test_infer_layers_with_embedding() {
        let names = vec!["model.embed_tokens.weight"];
        let layers = infer_layers_from_tensor_names(&names, None);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].name, "embedding");
    }

    #[test]
    fn test_infer_layers_with_wte() {
        let names = vec!["wte.weight"];
        let layers = infer_layers_from_tensor_names(&names, None);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].name, "embedding");
    }

    #[test]
    fn test_infer_layers_with_lm_head() {
        let names = vec!["lm_head.weight"];
        let layers = infer_layers_from_tensor_names(&names, None);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].name, "lm_head");
    }

    #[test]
    fn test_infer_layers_with_output_tensor() {
        let names = vec!["output.weight"];
        let layers = infer_layers_from_tensor_names(&names, None);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].name, "lm_head");
    }

    #[test]
    fn test_infer_layers_full_model() {
        let names = vec![
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.2.mlp.gate_proj.weight",
            "lm_head.weight",
        ];
        let layers = infer_layers_from_tensor_names(&names, None);
        // Should have: embedding + 3 transformer blocks + lm_head = 5
        assert_eq!(layers.len(), 5);
        assert_eq!(layers[0].name, "embedding");
        assert_eq!(layers[1].name, "transformer_block_0");
        assert_eq!(layers[2].name, "transformer_block_1");
        assert_eq!(layers[3].name, "transformer_block_2");
        assert_eq!(layers[4].name, "lm_head");
    }

    #[test]
    fn test_infer_layers_with_filter_matching() {
        let names = vec![
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.weight",
            "model.layers.1.self_attn.weight",
            "lm_head.weight",
        ];
        let layers = infer_layers_from_tensor_names(&names, Some("block_1"));
        // Filter should only include block_1
        assert!(layers.iter().any(|l| l.name == "transformer_block_1"));
        // Should not include block_0
        assert!(!layers.iter().any(|l| l.name == "transformer_block_0"));
    }

    #[test]
    fn test_infer_layers_with_filter_embedding() {
        let names = vec!["model.embed_tokens.weight", "model.layers.0.weight"];
        let layers = infer_layers_from_tensor_names(&names, Some("embedding"));
        assert!(layers.iter().any(|l| l.name == "embedding"));
    }

    #[test]
    fn test_infer_layers_with_filter_no_match() {
        let names = vec!["model.layers.0.weight", "model.layers.1.weight"];
        let layers = infer_layers_from_tensor_names(&names, Some("nonexistent"));
        assert!(layers.is_empty());
    }

    #[test]
    fn test_infer_layers_sorted_indices() {
        let names = vec![
            "model.layers.5.weight",
            "model.layers.0.weight",
            "model.layers.3.weight",
        ];
        let layers = infer_layers_from_tensor_names(&names, None);
        // BTreeMap ensures sorted order
        assert_eq!(layers[0].name, "transformer_block_0");
        assert_eq!(layers[1].name, "transformer_block_3");
        assert_eq!(layers[2].name, "transformer_block_5");
    }

    // ========================================================================
    // create_* helper function tests
    // ========================================================================

    #[test]
    fn test_create_embedding_layer() {
        let layer = create_embedding_layer(768);
        assert_eq!(layer.name, "embedding");
        assert_eq!(layer.index, None);
        assert!(layer.anomalies.is_empty());
        let output = layer.output_stats.expect("should have output stats");
        assert_eq!(output.count, 768);
    }

    #[test]
    fn test_create_embedding_layer_zero_dim() {
        let layer = create_embedding_layer(0);
        let output = layer.output_stats.expect("should have output stats");
        assert_eq!(output.count, 0);
    }

    #[test]
    fn test_create_final_layer_norm() {
        let layer = create_final_layer_norm();
        assert_eq!(layer.name, "final_layer_norm");
        assert_eq!(layer.index, None);
        assert!(layer.input_stats.is_none());
        assert!(layer.output_stats.is_none());
        assert!(layer.weight_stats.is_none());
        assert!(layer.anomalies.is_empty());
    }

    #[test]
    fn test_create_default_layer() {
        let layer = create_default_layer();
        assert!(layer.name.contains("not available"));
        assert_eq!(layer.index, None);
        assert_eq!(layer.anomalies.len(), 1);
        assert!(layer.anomalies[0].contains("No layer information"));
    }

    // ========================================================================
    // create_transformer_layers tests
    // ========================================================================

    #[test]
    fn test_create_transformer_layers_zero() {
        let layers = create_transformer_layers(0, None);
        assert!(layers.is_empty());
    }

    #[test]
    fn test_create_transformer_layers_basic() {
        let layers = create_transformer_layers(3, None);
        assert_eq!(layers.len(), 3);
        assert_eq!(layers[0].name, "transformer_block_0");
        assert_eq!(layers[0].index, Some(0));
        assert_eq!(layers[1].name, "transformer_block_1");
        assert_eq!(layers[1].index, Some(1));
        assert_eq!(layers[2].name, "transformer_block_2");
        assert_eq!(layers[2].index, Some(2));
    }

    #[test]
    fn test_create_transformer_layers_with_filter() {
        let layers = create_transformer_layers(10, Some("block_5"));
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].name, "transformer_block_5");
    }

    #[test]
    fn test_create_transformer_layers_filter_no_match() {
        let layers = create_transformer_layers(3, Some("nonexistent"));
        assert!(layers.is_empty());
    }

    #[test]
    fn test_create_transformer_layers_filter_multiple_match() {
        // Filter "block_1" matches "transformer_block_1" and "transformer_block_10" etc.
        let layers = create_transformer_layers(15, Some("block_1"));
        // Matches: block_1, block_10, block_11, block_12, block_13, block_14
        assert!(layers.len() >= 1);
        assert!(layers.iter().any(|l| l.name == "transformer_block_1"));
    }

    // ========================================================================
    // compute_trace_summary tests
    // ========================================================================

    #[test]
    fn test_compute_trace_summary_empty() {
        let summary = compute_trace_summary(&[], 0);
        assert_eq!(summary.total_layers, 0);
        assert_eq!(summary.total_parameters, 0);
        assert_eq!(summary.anomaly_count, 0);
        assert!(summary.anomalies.is_empty());
    }

    #[test]
    fn test_compute_trace_summary_no_anomalies() {
        let layers = vec![
            LayerTrace {
                name: "layer_0".to_string(),
                index: Some(0),
                input_stats: None,
                output_stats: None,
                weight_stats: None,
                anomalies: vec![],
            },
            LayerTrace {
                name: "layer_1".to_string(),
                index: Some(1),
                input_stats: None,
                output_stats: None,
                weight_stats: None,
                anomalies: vec![],
            },
        ];
        let summary = compute_trace_summary(&layers, 1000);
        assert_eq!(summary.total_layers, 2);
        assert_eq!(summary.total_parameters, 1000);
        assert_eq!(summary.anomaly_count, 0);
    }

    #[test]
    fn test_compute_trace_summary_with_anomalies() {
        let layers = vec![
            LayerTrace {
                name: "layer_0".to_string(),
                index: Some(0),
                input_stats: None,
                output_stats: None,
                weight_stats: None,
                anomalies: vec!["NaN detected".to_string()],
            },
            LayerTrace {
                name: "layer_1".to_string(),
                index: Some(1),
                input_stats: None,
                output_stats: None,
                weight_stats: None,
                anomalies: vec!["Inf detected".to_string(), "Large mean".to_string()],
            },
        ];
        let summary = compute_trace_summary(&layers, 5000);
        assert_eq!(summary.total_layers, 2);
        assert_eq!(summary.total_parameters, 5000);
        assert_eq!(summary.anomaly_count, 3);
        assert_eq!(summary.anomalies.len(), 3);
    }

    // ========================================================================
    // extract_layer_count / extract_model_dimension tests
    // ========================================================================

    #[test]
    fn test_extract_layer_count_n_layer() {
        let mut hp = serde_json::Map::new();
        hp.insert("n_layer".to_string(), serde_json::json!(32));
        assert_eq!(extract_layer_count(&hp), 32);
    }

    #[test]
    fn test_extract_layer_count_n_layers() {
        let mut hp = serde_json::Map::new();
        hp.insert("n_layers".to_string(), serde_json::json!(24));
        assert_eq!(extract_layer_count(&hp), 24);
    }

    #[test]
    fn test_extract_layer_count_missing() {
        let hp = serde_json::Map::new();
        assert_eq!(extract_layer_count(&hp), 0);
    }

    #[test]
    fn test_extract_layer_count_prefers_n_layer() {
        let mut hp = serde_json::Map::new();
        hp.insert("n_layer".to_string(), serde_json::json!(32));
        hp.insert("n_layers".to_string(), serde_json::json!(24));
        // n_layer is checked first
        assert_eq!(extract_layer_count(&hp), 32);
    }

    #[test]
    fn test_extract_model_dimension_n_embd() {
        let mut hp = serde_json::Map::new();
        hp.insert("n_embd".to_string(), serde_json::json!(768));
        assert_eq!(extract_model_dimension(&hp), 768);
    }

    #[test]
    fn test_extract_model_dimension_d_model() {
        let mut hp = serde_json::Map::new();
        hp.insert("d_model".to_string(), serde_json::json!(512));
        assert_eq!(extract_model_dimension(&hp), 512);
    }

    #[test]
    fn test_extract_model_dimension_missing() {
        let hp = serde_json::Map::new();
        assert_eq!(extract_model_dimension(&hp), 0);
    }

    #[test]
    fn test_extract_model_dimension_prefers_n_embd() {
        let mut hp = serde_json::Map::new();
        hp.insert("n_embd".to_string(), serde_json::json!(768));
        hp.insert("d_model".to_string(), serde_json::json!(512));
        assert_eq!(extract_model_dimension(&hp), 768);
    }

    // ========================================================================
    // extract_layers_from_hyperparameters tests
    // ========================================================================

    #[test]
    fn test_extract_layers_from_hyperparameters_basic() {
        let mut hp = serde_json::Map::new();
        hp.insert("n_layer".to_string(), serde_json::json!(3));
        hp.insert("n_embd".to_string(), serde_json::json!(256));

        let layers = extract_layers_from_hyperparameters(&hp, None);
        // embedding + 3 transformer blocks + final_layer_norm = 5
        assert_eq!(layers.len(), 5);
        assert_eq!(layers[0].name, "embedding");
        assert_eq!(layers[1].name, "transformer_block_0");
        assert_eq!(layers[2].name, "transformer_block_1");
        assert_eq!(layers[3].name, "transformer_block_2");
        assert_eq!(layers[4].name, "final_layer_norm");
    }

    #[test]
    fn test_extract_layers_from_hyperparameters_zero_layers() {
        let mut hp = serde_json::Map::new();
        hp.insert("n_embd".to_string(), serde_json::json!(256));
        // no n_layer key → 0 layers

        let layers = extract_layers_from_hyperparameters(&hp, None);
        // embedding + 0 transformer + final_layer_norm = 2
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].name, "embedding");
        assert_eq!(layers[1].name, "final_layer_norm");
    }

    #[test]
    fn test_extract_layers_from_hyperparameters_with_filter() {
        let mut hp = serde_json::Map::new();
        hp.insert("n_layer".to_string(), serde_json::json!(5));
        hp.insert("n_embd".to_string(), serde_json::json!(256));

        let layers = extract_layers_from_hyperparameters(&hp, Some("block_3"));
        // Only transformer_block_3 should match the filter + embedding + final_layer_norm
        assert!(layers.iter().any(|l| l.name == "transformer_block_3"));
    }

    // ========================================================================
    // handle_special_modes tests
    // ========================================================================

    #[test]
    fn test_handle_special_modes_interactive() {
        let path = Path::new("/tmp/model.apr");
        let result = handle_special_modes(path, None, false, false, true);
        assert!(result.is_some());
        assert!(result.expect("should be Some").is_ok());
    }

    #[test]
    fn test_handle_special_modes_diff_without_reference() {
        let path = Path::new("/tmp/model.apr");
        // diff mode without reference just prints a message and returns None
        let result = handle_special_modes(path, None, false, true, false);
        assert!(result.is_none());
    }

    #[test]
    fn test_handle_special_modes_diff_with_reference() {
        let path = Path::new("/tmp/model.apr");
        let ref_path = Path::new("/tmp/ref.apr");
        // diff mode with reference prints message and returns None (not handled here)
        let result = handle_special_modes(path, Some(ref_path), false, true, false);
        assert!(result.is_none());
    }

    #[test]
    fn test_handle_special_modes_none() {
        let path = Path::new("/tmp/model.apr");
        let result = handle_special_modes(path, None, false, false, false);
        assert!(result.is_none());
    }

    // ========================================================================
    // gguf_meta_u32 tests
    // ========================================================================

    #[test]
    fn test_gguf_meta_u32_uint32() {
        use aprender::format::gguf::GgufValue;
        let mut metadata = BTreeMap::new();
        metadata.insert("test.count".to_string(), GgufValue::Uint32(42));
        assert_eq!(gguf_meta_u32(&metadata, "test.count"), Some(42));
    }

    #[test]
    fn test_gguf_meta_u32_uint64() {
        use aprender::format::gguf::GgufValue;
        let mut metadata = BTreeMap::new();
        metadata.insert("test.count".to_string(), GgufValue::Uint64(100));
        assert_eq!(gguf_meta_u32(&metadata, "test.count"), Some(100));
    }

    #[test]
    fn test_gguf_meta_u32_int32() {
        use aprender::format::gguf::GgufValue;
        let mut metadata = BTreeMap::new();
        metadata.insert("test.count".to_string(), GgufValue::Int32(55));
        assert_eq!(gguf_meta_u32(&metadata, "test.count"), Some(55));
    }

    #[test]
    fn test_gguf_meta_u32_string_returns_none() {
        use aprender::format::gguf::GgufValue;
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "test.name".to_string(),
            GgufValue::String("hello".to_string()),
        );
        assert_eq!(gguf_meta_u32(&metadata, "test.name"), None);
    }

    #[test]
    fn test_gguf_meta_u32_missing_key() {
        let metadata: BTreeMap<String, aprender::format::gguf::GgufValue> = BTreeMap::new();
        assert_eq!(gguf_meta_u32(&metadata, "nonexistent"), None);
    }
