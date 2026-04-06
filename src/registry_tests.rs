//! Additional coverage tests for ModelRegistry - Part 2
//!
//! Focus: concurrent operations, edge cases, stress tests

#![allow(clippy::many_single_char_names)]

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use crate::error::RealizarError;
    use crate::layers::{Model, ModelConfig};
    use crate::registry::{ModelInfo, ModelRegistry};
    use crate::tokenizer::BPETokenizer;

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
                    format!("t{i}")
                }
            })
            .collect();
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");
        (model, tokenizer)
    }

    // ========================================================================
    // Concurrent Write Race Conditions
    // ========================================================================

    #[test]
    fn test_concurrent_register_same_id_race() {
        let registry = Arc::new(ModelRegistry::new(10));
        let success = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let errors = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let r = Arc::clone(&registry);
                let s = Arc::clone(&success);
                let e = Arc::clone(&errors);
                thread::spawn(move || {
                    let (m, t) = create_test_model();
                    match r.register("contested-id", m, t) {
                        Ok(()) => s.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
                        Err(RealizarError::ModelAlreadyExists(_)) => {
                            e.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                        },
                        Err(e) => panic!("Unexpected: {:?}", e),
                    };
                })
            })
            .collect();
        for h in handles {
            h.join().expect("join");
        }
        assert_eq!(success.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(errors.load(std::sync::atomic::Ordering::SeqCst), 9);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_concurrent_unregister_same_id_race() {
        let registry = Arc::new(ModelRegistry::new(10));
        let (m, t) = create_test_model();
        registry.register("to-remove", m, t).expect("reg");

        let success = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let errors = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let r = Arc::clone(&registry);
                let s = Arc::clone(&success);
                let e = Arc::clone(&errors);
                thread::spawn(move || match r.unregister("to-remove") {
                    Ok(()) => s.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
                    Err(RealizarError::ModelNotFound(_)) => {
                        e.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                    },
                    Err(e) => panic!("Unexpected: {:?}", e),
                })
            })
            .collect();
        for h in handles {
            h.join().expect("join");
        }
        assert_eq!(success.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(errors.load(std::sync::atomic::Ordering::SeqCst), 9);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_concurrent_replace_all_succeed() {
        let registry = Arc::new(ModelRegistry::new(10));
        let (m, t) = create_test_model();
        registry.register("to-replace", m, t).expect("reg");

        let success = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let handles: Vec<_> = (0..5)
            .map(|_| {
                let r = Arc::clone(&registry);
                let s = Arc::clone(&success);
                thread::spawn(move || {
                    let (m, t) = create_test_model();
                    r.replace("to-replace", m, t).expect("replace");
                    s.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                })
            })
            .collect();
        for h in handles {
            h.join().expect("join");
        }
        assert_eq!(success.load(std::sync::atomic::Ordering::SeqCst), 5);
        assert_eq!(registry.len(), 1);
    }

    // ========================================================================
    // Lock-Free Read Stress Test
    // ========================================================================

    #[test]
    fn test_concurrent_reads_during_writes() {
        let registry = Arc::new(ModelRegistry::new(100));
        for i in 0..10 {
            let (m, t) = create_test_model();
            registry.register(&format!("m{i}"), m, t).expect("reg");
        }

        let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let barrier = Arc::new(std::sync::Barrier::new(6)); // 5 readers + 1 main

        let readers: Vec<_> = (0..5)
            .map(|_| {
                let r = Arc::clone(&registry);
                let run = Arc::clone(&running);
                let b = Arc::clone(&barrier);
                thread::spawn(move || {
                    b.wait(); // Wait for all threads to be ready
                    let mut count = 0;
                    while run.load(std::sync::atomic::Ordering::Relaxed) {
                        let _ = r.list();
                        let _ = r.len();
                        for i in 0..10 {
                            let _ = r.contains(&format!("m{i}"));
                        }
                        count += 1;
                    }
                    count
                })
            })
            .collect();

        barrier.wait(); // Release all readers simultaneously

        for i in 10..20 {
            let (m, t) = create_test_model();
            registry.register(&format!("m{i}"), m, t).expect("reg");
            thread::sleep(Duration::from_micros(100)); // Increased from 50µs
        }

        running.store(false, std::sync::atomic::Ordering::Relaxed);
        let total: usize = readers.into_iter().map(|h| h.join().expect("join")).sum();
        assert!(total > 0);
        assert_eq!(registry.len(), 20);
    }

    // ========================================================================
    // Edge Cases: Unicode, Empty, Long IDs
    // ========================================================================

    #[test]
    fn test_unicode_and_special_ids() {
        let registry = ModelRegistry::new(5);

        // Unicode ID
        let (m1, t1) = create_test_model();
        let unicode_id = "\u{4e2d}\u{6587}-model";
        registry.register(unicode_id, m1, t1).expect("unicode");
        assert!(registry.contains(unicode_id));
        let info = registry.get_info(unicode_id).expect("info");
        assert_eq!(info.id, unicode_id);

        // Empty string ID
        let (m2, t2) = create_test_model();
        registry.register("", m2, t2).expect("empty");
        assert!(registry.contains(""));

        // Very long ID
        let (m3, t3) = create_test_model();
        let long_id: String = "x".repeat(5000);
        registry.register(&long_id, m3, t3).expect("long");
        assert!(registry.contains(&long_id));

        assert_eq!(registry.len(), 3);
    }

    #[test]
    fn test_model_info_unicode_fields() {
        let registry = ModelRegistry::new(5);
        let (m, t) = create_test_model();
        let info = ModelInfo {
            id: "unicode-info".to_string(),
            name: "Modelo \u{00f1}".to_string(),
            description: "\u{4e2d}\u{6587}".to_string(),
            format: "GGUF-\u{03b1}".to_string(),
            loaded: false,
        };
        registry.register_with_info(info, m, t).expect("reg");
        let ret = registry.get_info("unicode-info").expect("info");
        assert_eq!(ret.name, "Modelo \u{00f1}");
        assert_eq!(ret.description, "\u{4e2d}\u{6587}");
    }

    // ========================================================================
    // State Lifecycle Tests
    // ========================================================================

    #[test]
    fn test_register_unregister_cycle() {
        let registry = ModelRegistry::new(5);
        for cycle in 0..3 {
            let (m, t) = create_test_model();
            registry
                .register("cycle", m, t)
                .unwrap_or_else(|_| panic!("cycle {cycle}"));
            assert!(registry.contains("cycle"));
            registry
                .unregister("cycle")
                .unwrap_or_else(|_| panic!("unreg {cycle}"));
            assert!(!registry.contains("cycle"));
        }
    }

    #[test]
    fn test_replace_then_unregister() {
        let registry = ModelRegistry::new(5);
        let (m1, t1) = create_test_model();
        let (m2, t2) = create_test_model();
        registry.register("test", m1, t1).expect("reg");
        registry.replace("test", m2, t2).expect("replace");
        registry.unregister("test").expect("unreg");
        assert!(registry.is_empty());
    }

    #[test]
    fn test_arc_drops_after_unregister() {
        let registry = ModelRegistry::new(5);
        let (m, t) = create_test_model();
        registry.register("test", m, t).expect("reg");
        let (ma, ta) = registry.get("test").expect("get");
        assert!(Arc::strong_count(&ma) >= 2);
        registry.unregister("test").expect("unreg");
        assert_eq!(Arc::strong_count(&ma), 1);
        assert_eq!(Arc::strong_count(&ta), 1);
    }

    // ========================================================================
    // Additional Coverage: register_with_info race, contains edge cases
    // ========================================================================

    #[test]
    fn test_concurrent_register_with_info_race() {
        let registry = Arc::new(ModelRegistry::new(10));
        let success = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let errors = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let handles: Vec<_> = (0..5)
            .map(|i| {
                let r = Arc::clone(&registry);
                let s = Arc::clone(&success);
                let e = Arc::clone(&errors);
                thread::spawn(move || {
                    let (m, t) = create_test_model();
                    let info = ModelInfo {
                        id: "contested".to_string(),
                        name: format!("Thread {i}"),
                        description: String::new(),
                        format: "GGUF".to_string(),
                        loaded: false,
                    };
                    match r.register_with_info(info, m, t) {
                        Ok(()) => s.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
                        Err(RealizarError::ModelAlreadyExists(_)) => {
                            e.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                        },
                        Err(e) => panic!("Unexpected: {:?}", e),
                    };
                })
            })
            .collect();
        for h in handles {
            h.join().expect("join");
        }
        assert_eq!(success.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(errors.load(std::sync::atomic::Ordering::SeqCst), 4);
    }

    #[test]
    fn test_contains_nonexistent_various() {
        let registry = ModelRegistry::new(5);
        assert!(!registry.contains("nonexistent"));
        assert!(!registry.contains(""));
        assert!(!registry.contains(&"a".repeat(1000)));
        assert!(!registry.contains("\u{4e2d}\u{6587}"));
        assert!(!registry.contains("model with spaces"));
    }

    #[test]
    fn test_stress_many_models() {
        let registry = ModelRegistry::new(100);
        for i in 0..50 {
            let (m, t) = create_test_model();
            registry.register(&format!("s{i}"), m, t).expect("reg");
        }
        assert_eq!(registry.len(), 50);

        for i in 0..50 {
            assert!(registry.contains(&format!("s{i}")));
            let _ = registry.get(&format!("s{i}")).expect("get");
        }

        for i in 0..25 {
            registry.unregister(&format!("s{i}")).expect("unreg");
        }
        assert_eq!(registry.len(), 25);
        for i in 25..50 {
            assert!(registry.contains(&format!("s{i}")));
        }
    }

    #[test]
    fn test_duplicate_preserves_original_info() {
        let registry = ModelRegistry::new(5);
        let (m1, t1) = create_test_model();
        let (m2, t2) = create_test_model();
        let info1 = ModelInfo {
            id: "same".to_string(),
            name: "Original".to_string(),
            description: "First".to_string(),
            format: "GGUF".to_string(),
            loaded: false,
        };
        let info2 = ModelInfo {
            id: "same".to_string(),
            name: "New".to_string(),
            description: "Second".to_string(),
            format: "ST".to_string(),
            loaded: false,
        };
        registry.register_with_info(info1, m1, t1).expect("first");
        assert!(registry.register_with_info(info2, m2, t2).is_err());
        let ret = registry.get_info("same").expect("info");
        assert_eq!(ret.name, "Original");
        assert_eq!(ret.format, "GGUF");
    }

    #[test]
    fn test_list_returns_independent_clones() {
        let registry = ModelRegistry::new(5);
        let (m, t) = create_test_model();
        registry.register("test", m, t).expect("reg");
        let list1 = registry.list();
        let list2 = registry.list();
        assert_eq!(list1.len(), 1);
        assert_eq!(list2.len(), 1);
        assert_eq!(list1[0].id, list2[0].id);
    }

    #[test]
    fn test_list_order_independence() {
        let registry = ModelRegistry::new(10);
        for i in 0..5 {
            let (m, t) = create_test_model();
            registry.register(&format!("m{i}"), m, t).expect("reg");
        }
        let list = registry.list();
        let ids: std::collections::HashSet<_> = list.iter().map(|m| m.id.clone()).collect();
        for i in 0..5 {
            assert!(ids.contains(&format!("m{i}")));
        }
    }
}
