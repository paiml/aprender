//! Tests for `OwnedQuantizedModelCachedSync`
//!
//! Coverage for thread-safe cached model wrapper with Mutex-based scheduler caching.

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use crate::error::RealizarError;
    use crate::gguf::inference::cached::sync::OwnedQuantizedModelCachedSync;
    use crate::gguf::test_helpers::create_test_model_with_config;
    use crate::gguf::{BatchGenerationStats, GGUFConfig, QuantizedGenerateConfig};

    // ========================================================================
    // Batch Generation Stats Struct Tests
    // ========================================================================

    #[test]
    fn test_batch_generation_stats_fields() {
        let stats = BatchGenerationStats {
            gpu_cache_ready: true,
            cache_memory_gb: 1.5,
            num_layers: 32,
            hidden_dim: 2560,
            intermediate_dim: 10240,
            recommended_batch_size: 32,
            max_batch_size: 64,
        };

        assert!(stats.gpu_cache_ready);
        assert!((stats.cache_memory_gb - 1.5).abs() < f64::EPSILON);
        assert_eq!(stats.num_layers, 32);
        assert_eq!(stats.hidden_dim, 2560);
        assert_eq!(stats.intermediate_dim, 10240);
        assert_eq!(stats.recommended_batch_size, 32);
        assert_eq!(stats.max_batch_size, 64);
    }

    #[test]
    fn test_batch_generation_stats_clone() {
        let stats = BatchGenerationStats {
            gpu_cache_ready: false,
            cache_memory_gb: 0.0,
            num_layers: 1,
            hidden_dim: 64,
            intermediate_dim: 128,
            recommended_batch_size: 32,
            max_batch_size: 64,
        };

        let cloned = stats.clone();
        assert_eq!(cloned.gpu_cache_ready, stats.gpu_cache_ready);
        assert_eq!(cloned.num_layers, stats.num_layers);
    }

    #[test]
    fn test_batch_generation_stats_debug() {
        let stats = BatchGenerationStats {
            gpu_cache_ready: true,
            cache_memory_gb: 2.0,
            num_layers: 16,
            hidden_dim: 512,
            intermediate_dim: 2048,
            recommended_batch_size: 32,
            max_batch_size: 64,
        };

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("gpu_cache_ready"));
        assert!(debug_str.contains("16"));
    }
    include!("sync_tests_constructors.rs");
}
