//! Phase 34: Thread-safe cached sync.rs coverage tests
//!
//! These lib tests illuminate gguf/inference/cached/sync.rs:
//! - OwnedQuantizedModelCachedSync creation
//! - Thread-safe scheduler access
//! - Mutex-based interior mutability
//!
//! Located in lib tests to be included in `cargo test --lib` coverage

use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::GGUFConfig;

#[cfg(feature = "gpu")]
use crate::gguf::OwnedQuantizedModelCachedSync;

// =============================================================================
// OwnedQuantizedModelCachedSync Tests
// =============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_phase34_cached_sync_new() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let cached_sync = OwnedQuantizedModelCachedSync::new(model);

    // Verify we can access the inner model
    assert_eq!(cached_sync.model().config.hidden_dim, 64);
    assert_eq!(cached_sync.model().config.num_layers, 1);
}

#[test]
#[cfg(feature = "gpu")]
fn test_phase34_cached_sync_model_accessor() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 128,
        intermediate_dim: 256,
        num_layers: 2,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 1000,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let cached_sync = OwnedQuantizedModelCachedSync::new(model);

    // Access model multiple times - should be safe
    let m1 = cached_sync.model();
    let m2 = cached_sync.model();
    assert_eq!(m1.config.vocab_size, m2.config.vocab_size);
    assert_eq!(m1.config.num_layers, 2);
}

#[test]
#[cfg(feature = "gpu")]
fn test_phase34_cached_sync_concurrent_model_access() {
    use std::sync::Arc;
    use std::thread;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let cached_sync = Arc::new(OwnedQuantizedModelCachedSync::new(model));

    // Spawn multiple threads that access the model concurrently
    let mut handles = vec![];
    for i in 0..4 {
        let sync_clone = Arc::clone(&cached_sync);
        let handle = thread::spawn(move || {
            // Each thread accesses the model
            let m = sync_clone.model();
            assert_eq!(m.config.hidden_dim, 64);
            i * 10 // Return something to verify thread completed
        });
        handles.push(handle);
    }

    // Wait for all threads and verify they completed
    let results: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("thread join failed"))
        .collect();
    assert_eq!(results, vec![0, 10, 20, 30]);
}

#[test]
#[cfg(feature = "gpu")]
fn test_phase34_cached_sync_send_sync_bounds() {
    use std::sync::Arc;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let cached_sync = OwnedQuantizedModelCachedSync::new(model);

    // Verify Send + Sync by wrapping in Arc and sending to thread
    fn assert_send_sync<T: Send + Sync>(_: &T) {}
    assert_send_sync(&cached_sync);

    // Actually use it in a thread to verify Send
    let arc_sync = Arc::new(cached_sync);
    let arc_clone = Arc::clone(&arc_sync);
    let handle = std::thread::spawn(move || arc_clone.model().config.hidden_dim);
    assert_eq!(handle.join().expect("thread join failed"), 64);
}

#[test]
#[cfg(feature = "gpu")]
fn test_phase34_cached_sync_multiple_configs() {
    // Test with different architectures
    let configs = vec![
        ("llama", 64, 128, 1, 4, 4),
        ("qwen2", 128, 256, 2, 8, 4),
        ("phi3", 256, 512, 4, 16, 16),
    ];

    for (arch, hidden, intermediate, layers, heads, kv_heads) in configs {
        let config = GGUFConfig {
            architecture: arch.to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
            hidden_dim: hidden,
            intermediate_dim: intermediate,
            num_layers: layers,
            num_heads: heads,
            num_kv_heads: kv_heads,
            vocab_size: 100,
            context_length: 256,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: if arch == "qwen2" || arch == "phi3" {
                2
            } else {
                0
            },
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let model = create_test_model_with_config(&config);
        let cached_sync = OwnedQuantizedModelCachedSync::new(model);

        assert_eq!(cached_sync.model().config.architecture, arch);
        assert_eq!(cached_sync.model().config.hidden_dim, hidden);
        assert_eq!(cached_sync.model().config.num_layers, layers);
    }
}

// =============================================================================
// GPU Feature Disabled Tests (compile without gpu feature)
// =============================================================================

#[test]
#[cfg(not(feature = "gpu"))]
fn test_phase34_cached_sync_no_gpu_feature() {
    // When GPU feature is disabled, OwnedQuantizedModelCachedSync doesn't exist
    // This test just verifies compilation works without the feature
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let _model = create_test_model_with_config(&config);
    // Can't create CachedSync without GPU feature, but model creation works
}

// =============================================================================
// Stress Tests
// =============================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_phase34_cached_sync_rapid_access() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let cached_sync = OwnedQuantizedModelCachedSync::new(model);

    // Rapidly access the model many times
    for _ in 0..1000 {
        let _ = cached_sync.model().config.hidden_dim;
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_phase34_cached_sync_thread_stress() {
    use std::sync::Arc;
    use std::thread;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let cached_sync = Arc::new(OwnedQuantizedModelCachedSync::new(model));

    // Spawn many threads doing concurrent access
    let mut handles = vec![];
    for _ in 0..16 {
        let sync_clone = Arc::clone(&cached_sync);
        let handle = thread::spawn(move || {
            for _ in 0..100 {
                let _ = sync_clone.model().config.hidden_dim;
            }
        });
        handles.push(handle);
    }

    // All threads should complete without panic
    for handle in handles {
        handle.join().expect("Thread should not panic");
    }
}
