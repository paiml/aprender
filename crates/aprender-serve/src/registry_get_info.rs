
#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::ModelConfig;

    #[test]
    fn test_get_info_error_message() {
        let registry = ModelRegistry::new(5);
        let result = registry.get_info("another-nonexistent");

        match result {
            Err(RealizarError::ModelNotFound(id)) => {
                assert_eq!(id, "another-nonexistent");
            },
            _ => panic!("Expected ModelNotFound error"),
        }
    }

    #[test]
    fn test_unregister_error_message() {
        let registry = ModelRegistry::new(5);
        let result = registry.unregister("missing-model");

        match result {
            Err(RealizarError::ModelNotFound(id)) => {
                assert_eq!(id, "missing-model");
            },
            _ => panic!("Expected ModelNotFound error"),
        }
    }

    #[test]
    fn test_replace_error_message() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();
        let result = registry.replace("nonexistent", model, tokenizer);

        match result {
            Err(RealizarError::ModelNotFound(id)) => {
                assert_eq!(id, "nonexistent");
            },
            _ => panic!("Expected ModelNotFound error"),
        }
    }

    #[test]
    fn test_register_error_message() {
        let registry = ModelRegistry::new(5);
        let (model1, tokenizer1) = create_test_model();
        let (model2, tokenizer2) = create_test_model();

        registry
            .register("duplicate", model1, tokenizer1)
            .expect("first registration");

        let result = registry.register("duplicate", model2, tokenizer2);
        match result {
            Err(RealizarError::ModelAlreadyExists(id)) => {
                assert_eq!(id, "duplicate");
            },
            _ => panic!("Expected ModelAlreadyExists error"),
        }
    }

    #[test]
    fn test_concurrent_reads() {
        use std::thread;

        let registry = Arc::new(ModelRegistry::new(10));
        let (model, tokenizer) = create_test_model();
        registry
            .register("shared-model", model, tokenizer)
            .expect("registration");

        let mut handles = vec![];

        // Multiple concurrent reads
        for _ in 0..10 {
            let registry_clone = Arc::clone(&registry);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    let result = registry_clone.get("shared-model");
                    assert!(result.is_ok());
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("thread join");
        }
    }

    #[test]
    fn test_concurrent_list() {
        use std::thread;

        let registry = Arc::new(ModelRegistry::new(10));
        let (model, tokenizer) = create_test_model();
        registry
            .register("model", model, tokenizer)
            .expect("registration");

        let mut handles = vec![];

        // Multiple concurrent list calls
        for _ in 0..5 {
            let registry_clone = Arc::clone(&registry);
            let handle = thread::spawn(move || {
                for _ in 0..50 {
                    let list = registry_clone.list();
                    assert_eq!(list.len(), 1);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("thread join");
        }
    }

    #[test]
    fn test_new_with_various_capacities() {
        // Test with minimum capacity
        let registry = ModelRegistry::new(1);
        assert!(registry.is_empty());

        // Test with zero capacity (edge case)
        let registry = ModelRegistry::new(0);
        assert!(registry.is_empty());

        // Test with large capacity
        let registry = ModelRegistry::new(1000);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_register_default_info_values() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        registry
            .register("simple-model", model, tokenizer)
            .expect("registration");

        let info = registry.get_info("simple-model").expect("get info");
        // Check default values set by register()
        assert_eq!(info.id, "simple-model");
        assert_eq!(info.name, "simple-model"); // Defaults to id
        assert_eq!(info.description, ""); // Empty by default
        assert_eq!(info.format, "unknown"); // Unknown by default
        assert!(info.loaded); // Should be true
    }

    #[test]
    fn test_register_with_info_sets_loaded_true() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        // Pass loaded=false, but it should be set to true
        let info = ModelInfo {
            id: "test".to_string(),
            name: "Test".to_string(),
            description: "Desc".to_string(),
            format: "GGUF".to_string(),
            loaded: false, // Explicitly false
        };

        registry
            .register_with_info(info, model, tokenizer)
            .expect("registration");

        let retrieved = registry.get_info("test").expect("get info");
        assert!(retrieved.loaded); // Should be true despite passing false
    }

    #[test]
    fn test_contains_after_unregister() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        registry
            .register("test-model", model, tokenizer)
            .expect("registration");
        assert!(registry.contains("test-model"));

        registry.unregister("test-model").expect("unregister");
        assert!(!registry.contains("test-model"));
    }

    #[test]
    fn test_len_after_multiple_operations() {
        let registry = ModelRegistry::new(10);
        assert_eq!(registry.len(), 0);

        // Add 3 models
        for i in 0..3 {
            let (model, tokenizer) = create_test_model();
            registry
                .register(&format!("model-{i}"), model, tokenizer)
                .expect("registration");
        }
        assert_eq!(registry.len(), 3);

        // Remove 1 model
        registry.unregister("model-1").expect("unregister");
        assert_eq!(registry.len(), 2);

        // Replace (shouldn't change count)
        let (model, tokenizer) = create_test_model();
        registry
            .replace("model-0", model, tokenizer)
            .expect("replace");
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_model_info_empty_strings() {
        let info = ModelInfo {
            id: String::new(),
            name: String::new(),
            description: String::new(),
            format: String::new(),
            loaded: false,
        };

        assert!(info.id.is_empty());
        assert!(info.name.is_empty());
        assert!(info.description.is_empty());
        assert!(info.format.is_empty());
        assert!(!info.loaded);
    }

    #[test]
    fn test_model_info_special_characters() {
        let info = ModelInfo {
            id: "model/with:special@chars!".to_string(),
            name: "Name with spaces".to_string(),
            description: "Line1\nLine2\tTabbed".to_string(),
            format: "GGUF-v3.0".to_string(),
            loaded: true,
        };

        let json = serde_json::to_string(&info).expect("serialize");
        let deserialized: ModelInfo = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.id, info.id);
        assert_eq!(deserialized.name, info.name);
        assert_eq!(deserialized.description, info.description);
        assert_eq!(deserialized.format, info.format);
    }

    #[test]
    fn test_multiple_unregisters_different_models() {
        let registry = ModelRegistry::new(10);

        // Register 5 models
        for i in 0..5 {
            let (model, tokenizer) = create_test_model();
            registry
                .register(&format!("model-{i}"), model, tokenizer)
                .expect("registration");
        }
        assert_eq!(registry.len(), 5);

        // Unregister them all in reverse order
        for i in (0..5).rev() {
            registry
                .unregister(&format!("model-{i}"))
                .expect("unregister");
            assert_eq!(registry.len(), i);
        }
        assert!(registry.is_empty());
    }

    #[test]
    fn test_arc_reference_counts() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        registry
            .register("test", model, tokenizer)
            .expect("registration");

        // Get multiple references
        let (m1, t1) = registry.get("test").expect("get1");
        let (m2, t2) = registry.get("test").expect("get2");
        let (m3, t3) = registry.get("test").expect("get3");

        // All should point to the same underlying data
        assert!(Arc::ptr_eq(&m1, &m2));
        assert!(Arc::ptr_eq(&m2, &m3));
        assert!(Arc::ptr_eq(&t1, &t2));
        assert!(Arc::ptr_eq(&t2, &t3));

        // Strong count should be at least 4 (registry + 3 gets)
        assert!(Arc::strong_count(&m1) >= 4);
        assert!(Arc::strong_count(&t1) >= 4);
    }

    #[test]
    fn test_list_returns_cloned_info() {
        let registry = ModelRegistry::new(5);
        let (model, tokenizer) = create_test_model();

        let info = ModelInfo {
            id: "test".to_string(),
            name: "Test Model".to_string(),
            description: "Description".to_string(),
            format: "GGUF".to_string(),
            loaded: false,
        };

        registry
            .register_with_info(info, model, tokenizer)
            .expect("registration");

        let list1 = registry.list();
        let list2 = registry.list();

        // Lists should be independent copies
        assert_eq!(list1.len(), 1);
        assert_eq!(list2.len(), 1);
        assert_eq!(list1[0].id, list2[0].id);
    }
include!("registry_create.rs");
}
