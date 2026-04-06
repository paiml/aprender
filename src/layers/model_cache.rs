
#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // KVCache tests
    // ========================================================================

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::new(4, 128, 64).expect("cache");
        assert_eq!(cache.num_layers(), 4);
        assert_eq!(cache.max_seq_len(), 128);
        assert_eq!(cache.head_dim(), 64);
        assert_eq!(cache.current_pos(), 0);
    }

    #[test]
    fn test_kv_cache_new_zero_layers_error() {
        let result = KVCache::new(0, 128, 64);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("num_layers must be > 0"));
    }

    #[test]
    fn test_kv_cache_new_zero_seq_len_error() {
        let result = KVCache::new(4, 0, 64);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("max_seq_len must be > 0"));
    }

    #[test]
    fn test_kv_cache_new_zero_head_dim_error() {
        let result = KVCache::new(4, 128, 0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("head_dim must be > 0"));
    }

    #[test]
    fn test_kv_cache_update_and_advance() {
        let mut cache = KVCache::new(2, 4, 3).expect("cache");
        let key = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("key");
        let value = Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]).expect("value");

        cache.update(0, &key, &value).expect("update");
        cache.advance();

        assert_eq!(cache.current_pos(), 1);
    }

    #[test]
    fn test_kv_cache_update_layer_out_of_bounds() {
        let mut cache = KVCache::new(2, 4, 3).expect("cache");
        let key = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("key");
        let value = Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]).expect("value");

        let result = cache.update(5, &key, &value);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));
    }

    #[test]
    fn test_kv_cache_update_key_size_mismatch() {
        let mut cache = KVCache::new(2, 4, 3).expect("cache");
        let key = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("key"); // Wrong size
        let value = Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]).expect("value");

        let result = cache.update(0, &key, &value);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Key size"));
    }

    #[test]
    fn test_kv_cache_update_value_size_mismatch() {
        let mut cache = KVCache::new(2, 4, 3).expect("cache");
        let key = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("key");
        let value = Tensor::from_vec(vec![2], vec![4.0, 5.0]).expect("value"); // Wrong size

        let result = cache.update(0, &key, &value);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Value size"));
    }

    #[test]
    fn test_kv_cache_get_key_empty() {
        let cache = KVCache::new(2, 4, 3).expect("cache");
        let key = cache.get_key(0).expect("key");
        assert_eq!(key.shape(), &[1, 3]); // Returns [1, head_dim] when empty
    }

    #[test]
    fn test_kv_cache_get_value_empty() {
        let cache = KVCache::new(2, 4, 3).expect("cache");
        let value = cache.get_value(0).expect("value");
        assert_eq!(value.shape(), &[1, 3]);
    }

    #[test]
    fn test_kv_cache_get_key_after_update() {
        let mut cache = KVCache::new(2, 4, 3).expect("cache");
        let key = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("key");
        let value = Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]).expect("value");

        cache.update(0, &key, &value).expect("update");
        cache.advance();

        let cached_key = cache.get_key(0).expect("cached_key");
        assert_eq!(cached_key.shape(), &[1, 3]);
        let data = cached_key.data();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_kv_cache_get_value_after_update() {
        let mut cache = KVCache::new(2, 4, 3).expect("cache");
        let key = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("key");
        let value = Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]).expect("value");

        cache.update(0, &key, &value).expect("update");
        cache.advance();

        let cached_value = cache.get_value(0).expect("cached_value");
        let data = cached_value.data();
        assert!((data[0] - 4.0).abs() < 1e-6);
        assert!((data[1] - 5.0).abs() < 1e-6);
        assert!((data[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_kv_cache_get_key_layer_out_of_bounds() {
        let cache = KVCache::new(2, 4, 3).expect("cache");
        let result = cache.get_key(5);
        assert!(result.is_err());
    }

    #[test]
    fn test_kv_cache_get_value_layer_out_of_bounds() {
        let cache = KVCache::new(2, 4, 3).expect("cache");
        let result = cache.get_value(5);
        assert!(result.is_err());
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = KVCache::new(2, 4, 3).expect("cache");
        let key = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("key");
        let value = Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]).expect("value");

        cache.update(0, &key, &value).expect("update");
        cache.advance();
        assert_eq!(cache.current_pos(), 1);

        cache.clear();
        assert_eq!(cache.current_pos(), 0);
    }

    #[test]
    fn test_kv_cache_cache_full_error() {
        let mut cache = KVCache::new(1, 2, 2).expect("cache");
        let key = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("key");
        let value = Tensor::from_vec(vec![2], vec![3.0, 4.0]).expect("value");

        // Fill cache
        cache.update(0, &key, &value).expect("update");
        cache.advance();
        cache.update(0, &key, &value).expect("update");
        cache.advance();

        // Cache is full now
        let result = cache.update(0, &key, &value);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Cache full"));
    }

    #[test]
    fn test_kv_cache_multiple_layers() {
        let mut cache = KVCache::new(3, 4, 2).expect("cache");
        let key0 = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("key0");
        let val0 = Tensor::from_vec(vec![2], vec![3.0, 4.0]).expect("val0");
        let key1 = Tensor::from_vec(vec![2], vec![5.0, 6.0]).expect("key1");
        let val1 = Tensor::from_vec(vec![2], vec![7.0, 8.0]).expect("val1");
        let key2 = Tensor::from_vec(vec![2], vec![9.0, 10.0]).expect("key2");
        let val2 = Tensor::from_vec(vec![2], vec![11.0, 12.0]).expect("val2");

        cache.update(0, &key0, &val0).expect("update");
        cache.update(1, &key1, &val1).expect("update");
        cache.update(2, &key2, &val2).expect("update");
        cache.advance();

        let k0 = cache.get_key(0).expect("k0");
        let k1 = cache.get_key(1).expect("k1");
        let k2 = cache.get_key(2).expect("k2");

        assert!((k0.data()[0] - 1.0).abs() < 1e-6);
        assert!((k1.data()[0] - 5.0).abs() < 1e-6);
        assert!((k2.data()[0] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_kv_cache_clone() {
        let cache = KVCache::new(2, 4, 3).expect("cache");
        let cloned = cache.clone();
        assert_eq!(cloned.num_layers(), 2);
        assert_eq!(cloned.max_seq_len(), 4);
    }

    #[test]
    fn test_kv_cache_debug() {
        let cache = KVCache::new(2, 4, 3).expect("cache");
        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("KVCache"));
    }

    // ========================================================================
    // ModelConfig tests
    // ========================================================================

    #[test]
    fn test_model_config_new() {
        let config = ModelConfig {
            vocab_size: 32000,
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            intermediate_dim: 11008,
            eps: 1e-5,
        };
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_dim, 4096);
        assert_eq!(config.num_layers, 32);
    }

    #[test]
    fn test_model_config_clone() {
        let config = ModelConfig {
            vocab_size: 1000,
            hidden_dim: 256,
            num_layers: 4,
            num_heads: 4,
            intermediate_dim: 1024,
            eps: 1e-5,
        };
        let cloned = config.clone();
        assert_eq!(cloned.vocab_size, 1000);
    }

    #[test]
    fn test_model_config_debug() {
        let config = ModelConfig {
            vocab_size: 1000,
            hidden_dim: 256,
            num_layers: 4,
            num_heads: 4,
            intermediate_dim: 1024,
            eps: 1e-5,
        };
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("ModelConfig"));
        assert!(debug_str.contains("vocab_size: 1000"));
    }

    // ========================================================================
    // Embedding tests (basic structure)
    // ========================================================================

    #[test]
    fn test_embedding_new() {
        let embedding = Embedding::new(100, 64).expect("embedding");
        assert_eq!(embedding.vocab_size(), 100);
        assert_eq!(embedding.embed_dim(), 64);
    }

    #[test]
    fn test_embedding_new_zero_vocab_error() {
        let result = Embedding::new(0, 64);
        assert!(result.is_err());
    }

    #[test]
    fn test_embedding_new_zero_dim_error() {
        let result = Embedding::new(100, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_embedding_forward() {
        let embedding = Embedding::new(100, 4).expect("embedding");
        let indices: &[usize] = &[0, 1];
        let result = embedding.forward(indices).expect("result");
        // Shape should be [2, 4] for 2 tokens with dim 4
        assert_eq!(result.shape(), &[2, 4]);
    }

    #[test]
    fn test_embedding_forward_single_token() {
        let embedding = Embedding::new(50, 8).expect("embedding");
        let indices: &[usize] = &[5];
        let result = embedding.forward(indices).expect("result");
        assert_eq!(result.shape(), &[1, 8]);
    }

    #[test]
    fn test_embedding_forward_out_of_bounds() {
        let embedding = Embedding::new(10, 4).expect("embedding");
        let indices: &[usize] = &[100]; // Out of vocab
        let result = embedding.forward(indices);
        assert!(result.is_err());
    }

    #[test]
    fn test_embedding_clone() {
        let embedding = Embedding::new(100, 64).expect("embedding");
        let cloned = embedding.clone();
        assert_eq!(cloned.vocab_size(), 100);
    }
}
