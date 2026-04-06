
    fn create_test_model() -> (Model, BPETokenizer) {
        let config = ModelConfig {
            vocab_size: 100,
            hidden_dim: 32,
            num_heads: 1,
            num_layers: 1,
            intermediate_dim: 64,
            eps: 1e-5,
        };
        let model = Model::new(config).expect("test");

        let vocab: Vec<String> = (0..100)
            .map(|i| {
                if i == 0 {
                    "<unk>".to_string()
                } else {
                    format!("token{i}")
                }
            })
            .collect();
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        (model, tokenizer)
    }

    #[test]
    fn test_registry_creation() {
        let registry = ModelRegistry::new(5);
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_register_model() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        registry
            .register("test-model", model, tokenizer)
            .expect("test");

        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
        assert!(registry.contains("test-model"));
    }

    #[test]
    fn test_register_duplicate_error() {
        let registry = ModelRegistry::new(5);
        let (model1, tokenizer1) = create_test_model();
        let (model2, tokenizer2) = create_test_model();

        registry
            .register("test-model", model1, tokenizer1)
            .expect("test");
        let result = registry.register("test-model", model2, tokenizer2);

        assert!(result.is_err());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_get_model() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        registry
            .register("test-model", model, tokenizer)
            .expect("test");

        let (retrieved_model, retrieved_tokenizer) = registry.get("test-model").expect("test");
        assert!(Arc::strong_count(&retrieved_model) >= 2); // Registry + local
        assert!(Arc::strong_count(&retrieved_tokenizer) >= 2);
    }

    #[test]
    fn test_get_nonexistent_model() {
        let registry = ModelRegistry::new(5);
        let result = registry.get("nonexistent");

        assert!(result.is_err());
    }

    #[test]
    fn test_register_with_info() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        let info = ModelInfo {
            id: "llama-7b".to_string(),
            name: "Llama 7B".to_string(),
            description: "7B parameter Llama model".to_string(),
            format: "GGUF".to_string(),
            loaded: false,
        };

        registry
            .register_with_info(info.clone(), model, tokenizer)
            .expect("test");

        let retrieved_info = registry.get_info("llama-7b").expect("test");
        assert_eq!(retrieved_info.id, "llama-7b");
        assert_eq!(retrieved_info.name, "Llama 7B");
        assert_eq!(retrieved_info.description, "7B parameter Llama model");
        assert_eq!(retrieved_info.format, "GGUF");
        assert!(retrieved_info.loaded); // Should be set to true
    }

    #[test]
    fn test_list_models() {
        let registry = ModelRegistry::new(5);
        let (model1, tokenizer1) = create_test_model();
        let (model2, tokenizer2) = create_test_model();

        registry
            .register("model-1", model1, tokenizer1)
            .expect("test");
        registry
            .register("model-2", model2, tokenizer2)
            .expect("test");

        let model_list = registry.list();
        assert_eq!(model_list.len(), 2);

        let ids: Vec<String> = model_list.iter().map(|m| m.id.clone()).collect();
        assert!(ids.contains(&"model-1".to_string()));
        assert!(ids.contains(&"model-2".to_string()));
    }

    #[test]
    fn test_unregister_model() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        registry
            .register("test-model", model, tokenizer)
            .expect("test");
        assert_eq!(registry.len(), 1);

        registry.unregister("test-model").expect("test");
        assert_eq!(registry.len(), 0);
        assert!(!registry.contains("test-model"));
    }

    #[test]
    fn test_unregister_nonexistent() {
        let registry = ModelRegistry::new(5);
        let result = registry.unregister("nonexistent");

        assert!(result.is_err());
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let registry = Arc::new(ModelRegistry::new(10));
        let mut handles = vec![];

        // Register models from multiple threads
        for i in 0..5 {
            let registry_clone = Arc::clone(&registry);
            let handle = thread::spawn(move || {
                let (model, tokenizer) = create_test_model();
                registry_clone
                    .register(&format!("model-{i}"), model, tokenizer)
                    .expect("test");
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("test");
        }

        assert_eq!(registry.len(), 5);
    }

    #[test]
    fn test_multiple_get_same_model() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        registry
            .register("test-model", model, tokenizer)
            .expect("test");

        // Get the same model multiple times
        let (model1, _) = registry.get("test-model").expect("test");
        let (model2, _) = registry.get("test-model").expect("test");

        // Both should point to the same underlying model
        assert!(Arc::ptr_eq(&model1, &model2));
    }

    #[test]
    fn test_replace_model() {
        let registry = ModelRegistry::new(5);
        let (model1, tokenizer1) = create_test_model();
        let (model2, tokenizer2) = create_test_model();

        // Register initial model
        registry
            .register("test-model", model1, tokenizer1)
            .expect("test");
        assert_eq!(registry.len(), 1);

        // Replace with new model
        registry
            .replace("test-model", model2, tokenizer2)
            .expect("test");
        assert_eq!(registry.len(), 1);

        // Verify replacement worked
        let (retrieved, _) = registry.get("test-model").expect("test");
        assert!(Arc::strong_count(&retrieved) >= 2);
    }

    #[test]
    fn test_replace_nonexistent_model() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        // Try to replace a model that doesn't exist
        let result = registry.replace("nonexistent", model, tokenizer);
        assert!(result.is_err());
    }

    #[test]
    fn test_contains_method() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        assert!(!registry.contains("test-model"));
        registry
            .register("test-model", model, tokenizer)
            .expect("test");
        assert!(registry.contains("test-model"));
    }

    #[test]
    fn test_is_empty_method() {
        let registry = ModelRegistry::new(5);
        assert!(registry.is_empty());

        let (model, tokenizer) = create_test_model();
        registry
            .register("test-model", model, tokenizer)
            .expect("test");
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_get_info_nonexistent() {
        let registry = ModelRegistry::new(5);
        let result = registry.get_info("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_len_method() {
        let registry = ModelRegistry::new(5);
        assert_eq!(registry.len(), 0);

        let (model1, tokenizer1) = create_test_model();
        registry
            .register("model-1", model1, tokenizer1)
            .expect("test");
        assert_eq!(registry.len(), 1);

        let (model2, tokenizer2) = create_test_model();
        registry
            .register("model-2", model2, tokenizer2)
            .expect("test");
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_register_with_info_duplicate_error() {
        let registry = ModelRegistry::new(5);
        let (model1, tokenizer1) = create_test_model();
        let (model2, tokenizer2) = create_test_model();

        let info1 = ModelInfo {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            description: "A test model".to_string(),
            format: "GGUF".to_string(),
            loaded: false,
        };

        let info2 = ModelInfo {
            id: "test-model".to_string(),
            name: "Another Test".to_string(),
            description: "Same ID".to_string(),
            format: "Safetensors".to_string(),
            loaded: false,
        };

        registry
            .register_with_info(info1, model1, tokenizer1)
            .expect("first registration should succeed");

        let result = registry.register_with_info(info2, model2, tokenizer2);
        assert!(result.is_err());
        match result {
            Err(RealizarError::ModelAlreadyExists(id)) => {
                assert_eq!(id, "test-model");
            },
            _ => panic!("Expected ModelAlreadyExists error"),
        }
    }

    #[test]
    fn test_model_info_serialization() {
        let info = ModelInfo {
            id: "llama-7b".to_string(),
            name: "Llama 7B".to_string(),
            description: "7B parameter model".to_string(),
            format: "GGUF".to_string(),
            loaded: true,
        };

        // Test serialization roundtrip
        let json = serde_json::to_string(&info).expect("serialization");
        let deserialized: ModelInfo = serde_json::from_str(&json).expect("deserialization");

        assert_eq!(deserialized.id, info.id);
        assert_eq!(deserialized.name, info.name);
        assert_eq!(deserialized.description, info.description);
        assert_eq!(deserialized.format, info.format);
        assert_eq!(deserialized.loaded, info.loaded);
    }

    #[test]
    fn test_model_info_debug() {
        let info = ModelInfo {
            id: "test".to_string(),
            name: "Test".to_string(),
            description: "Desc".to_string(),
            format: "GGUF".to_string(),
            loaded: true,
        };

        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("test"));
        assert!(debug_str.contains("ModelInfo"));
    }

    #[test]
    fn test_model_info_clone() {
        let info = ModelInfo {
            id: "original".to_string(),
            name: "Original".to_string(),
            description: "Original desc".to_string(),
            format: "GGUF".to_string(),
            loaded: true,
        };

        let cloned = info.clone();
        assert_eq!(cloned.id, info.id);
        assert_eq!(cloned.name, info.name);
        assert_eq!(cloned.description, info.description);
        assert_eq!(cloned.format, info.format);
        assert_eq!(cloned.loaded, info.loaded);
    }

    #[test]
    fn test_replace_preserves_metadata() {
        let registry = ModelRegistry::new(5);
        let (model1, tokenizer1) = create_test_model();
        let (model2, tokenizer2) = create_test_model();

        let info = ModelInfo {
            id: "test-model".to_string(),
            name: "My Custom Name".to_string(),
            description: "My custom description".to_string(),
            format: "Safetensors".to_string(),
            loaded: false,
        };

        registry
            .register_with_info(info, model1, tokenizer1)
            .expect("registration");

        // Replace with new model
        registry
            .replace("test-model", model2, tokenizer2)
            .expect("replacement");

        // Verify metadata is preserved
        let retrieved_info = registry.get_info("test-model").expect("get info");
        assert_eq!(retrieved_info.name, "My Custom Name");
        assert_eq!(retrieved_info.description, "My custom description");
        assert_eq!(retrieved_info.format, "Safetensors");
    }

    #[test]
    fn test_list_empty_registry() {
        let registry = ModelRegistry::new(5);
        let list = registry.list();
        assert!(list.is_empty());
    }

    #[test]
    fn test_unregister_then_register_same_id() {
        let registry = ModelRegistry::new(5);
        let (model1, tokenizer1) = create_test_model();
        let (model2, tokenizer2) = create_test_model();

        // Register first model
        registry
            .register("test-model", model1, tokenizer1)
            .expect("first registration");

        // Unregister
        registry.unregister("test-model").expect("unregistration");
        assert!(!registry.contains("test-model"));

        // Register again with same ID
        registry
            .register("test-model", model2, tokenizer2)
            .expect("second registration");
        assert!(registry.contains("test-model"));
    }

    #[test]
    fn test_get_error_message() {
        let registry = ModelRegistry::new(5);
        let result = registry.get("nonexistent-model-xyz");

        match result {
            Err(RealizarError::ModelNotFound(id)) => {
                assert_eq!(id, "nonexistent-model-xyz");
            },
            _ => panic!("Expected ModelNotFound error"),
        }
    }
