use crate::generate::GenerationConfig;
use crate::layers::*;

#[test]
fn test_kvcache_creation() {
    let cache = KVCache::new(4, 512, 64).expect("test");
    assert_eq!(cache.num_layers(), 4);
    assert_eq!(cache.max_seq_len(), 512);
    assert_eq!(cache.head_dim(), 64);
    assert_eq!(cache.current_pos(), 0);
    assert!(!cache.is_full());
}

#[test]
fn test_kvcache_zero_params_error() {
    assert!(KVCache::new(0, 512, 64).is_err());
    assert!(KVCache::new(4, 0, 64).is_err());
    assert!(KVCache::new(4, 512, 0).is_err());
}

#[test]
fn test_kvcache_update_and_retrieve() {
    let mut cache = KVCache::new(2, 10, 4).expect("test");

    // Add first position
    let key = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let value = Tensor::from_vec(vec![4], vec![5.0, 6.0, 7.0, 8.0]).expect("test");

    cache.update(0, &key, &value).expect("test");
    cache.advance();

    // Retrieve and verify
    let cached_key = cache.get_key(0).expect("test");
    let cached_value = cache.get_value(0).expect("test");

    assert_eq!(cached_key.shape(), &[1, 4]);
    assert_eq!(cached_value.shape(), &[1, 4]);

    for i in 0..4 {
        assert!((cached_key.data()[i] - key.data()[i]).abs() < 1e-6);
        assert!((cached_value.data()[i] - value.data()[i]).abs() < 1e-6);
    }
}

#[test]
fn test_kvcache_multiple_positions() {
    let mut cache = KVCache::new(1, 10, 2).expect("test");

    // Add multiple positions
    for pos in 0..3 {
        #[allow(clippy::cast_precision_loss)]
        let base = pos as f32;
        let key = Tensor::from_vec(vec![2], vec![base, base + 0.5]).expect("test");
        let value = Tensor::from_vec(vec![2], vec![base + 1.0, base + 1.5]).expect("test");

        cache.update(0, &key, &value).expect("test");
        cache.advance();
    }

    assert_eq!(cache.current_pos(), 3);

    // Retrieve all positions
    let cached_key = cache.get_key(0).expect("test");
    let cached_value = cache.get_value(0).expect("test");

    assert_eq!(cached_key.shape(), &[3, 2]);
    assert_eq!(cached_value.shape(), &[3, 2]);

    // Verify first position
    assert!((cached_key.data()[0] - 0.0).abs() < 1e-6);
    assert!((cached_key.data()[1] - 0.5).abs() < 1e-6);
    // Verify second position
    assert!((cached_key.data()[2] - 1.0).abs() < 1e-6);
    assert!((cached_key.data()[3] - 1.5).abs() < 1e-6);
}

#[test]
fn test_kvcache_multiple_layers() {
    let mut cache = KVCache::new(2, 10, 4).expect("test");

    let key0 = Tensor::from_vec(vec![4], vec![1.0; 4]).expect("test");
    let value0 = Tensor::from_vec(vec![4], vec![2.0; 4]).expect("test");
    let key1 = Tensor::from_vec(vec![4], vec![3.0; 4]).expect("test");
    let value1 = Tensor::from_vec(vec![4], vec![4.0; 4]).expect("test");

    cache.update(0, &key0, &value0).expect("test");
    cache.update(1, &key1, &value1).expect("test");
    cache.advance();

    // Verify layer 0
    let layer0_key = cache.get_key(0).expect("test");
    assert!((layer0_key.data()[0] - 1.0).abs() < 1e-6);

    // Verify layer 1
    let layer1_key = cache.get_key(1).expect("test");
    assert!((layer1_key.data()[0] - 3.0).abs() < 1e-6);
}

#[test]
fn test_kvcache_layer_out_of_bounds_error() {
    let mut cache = KVCache::new(2, 10, 4).expect("test");
    let key = Tensor::from_vec(vec![4], vec![1.0; 4]).expect("test");
    let value = Tensor::from_vec(vec![4], vec![2.0; 4]).expect("test");

    // Update layer 2 (out of bounds)
    assert!(cache.update(2, &key, &value).is_err());

    // Get layer 2 (out of bounds)
    assert!(cache.get_key(2).is_err());
    assert!(cache.get_value(2).is_err());
}

#[test]
fn test_kvcache_size_mismatch_error() {
    let mut cache = KVCache::new(1, 10, 4).expect("test");

    // Wrong key size
    let key = Tensor::from_vec(vec![3], vec![1.0; 3]).expect("test");
    let value = Tensor::from_vec(vec![4], vec![2.0; 4]).expect("test");
    assert!(cache.update(0, &key, &value).is_err());

    // Wrong value size
    let key = Tensor::from_vec(vec![4], vec![1.0; 4]).expect("test");
    let value = Tensor::from_vec(vec![3], vec![2.0; 3]).expect("test");
    assert!(cache.update(0, &key, &value).is_err());
}

#[test]
fn test_kvcache_full_error() {
    let mut cache = KVCache::new(1, 2, 4).expect("test");
    let key = Tensor::from_vec(vec![4], vec![1.0; 4]).expect("test");
    let value = Tensor::from_vec(vec![4], vec![2.0; 4]).expect("test");

    // Fill cache
    cache.update(0, &key, &value).expect("test");
    cache.advance();
    cache.update(0, &key, &value).expect("test");
    cache.advance();

    assert!(cache.is_full());

    // Try to add more
    assert!(cache.update(0, &key, &value).is_err());
}

#[test]
fn test_kvcache_clear() {
    let mut cache = KVCache::new(1, 10, 4).expect("test");
    let key = Tensor::from_vec(vec![4], vec![1.0; 4]).expect("test");
    let value = Tensor::from_vec(vec![4], vec![2.0; 4]).expect("test");

    cache.update(0, &key, &value).expect("test");
    cache.advance();
    assert_eq!(cache.current_pos(), 1);

    cache.clear();
    assert_eq!(cache.current_pos(), 0);
    assert!(!cache.is_full());
}

#[test]
fn test_kvcache_empty_retrieval() {
    let cache = KVCache::new(1, 10, 4).expect("test");

    // Retrieve from empty cache
    let cached_key = cache.get_key(0).expect("test");
    let cached_value = cache.get_value(0).expect("test");

    // Should return [1, 4] tensor with zeros
    assert_eq!(cached_key.shape(), &[1, 4]);
    assert_eq!(cached_value.shape(), &[1, 4]);
    for &val in cached_key.data() {
        assert!((val - 0.0).abs() < 1e-6);
    }
}

// TransformerBlock tests

#[test]
fn test_transformer_block_creation() {
    let block = TransformerBlock::new(64, 4, 256, 1e-5).expect("test");
    assert_eq!(block.hidden_dim(), 64);
}

#[test]
fn test_transformer_block_zero_params_error() {
    // Zero hidden_dim
    assert!(TransformerBlock::new(0, 4, 256, 1e-5).is_err());
    // Zero num_heads
    assert!(TransformerBlock::new(64, 0, 256, 1e-5).is_err());
    // Zero intermediate_dim
    assert!(TransformerBlock::new(64, 4, 0, 1e-5).is_err());
}

#[test]
fn test_transformer_block_head_divisibility_error() {
    // 63 not divisible by 4
    assert!(TransformerBlock::new(63, 4, 256, 1e-5).is_err());
}

#[test]
fn test_transformer_block_forward_shape() {
    // Use num_heads=1 so head_dim=hidden_dim (simplified single-head attention)
    let block = TransformerBlock::new(8, 1, 32, 1e-5).expect("test");

    // Input: [2, 8] (seq_len=2, hidden_dim=8)
    let input = Tensor::from_vec(vec![2, 8], vec![0.1; 16]).expect("test");
    let output = block.forward(&input).expect("test");

    // Output should have same shape
    assert_eq!(output.shape(), &[2, 8]);
    assert_eq!(output.data().len(), 16);
}

#[test]
fn test_transformer_block_forward_valid_output() {
    let block = TransformerBlock::new(4, 1, 16, 1e-5).expect("test");

    // Input must be 2D [seq_len, hidden_dim]
    let input = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let output = block.forward(&input).expect("test");

    // Output should be finite
    for &val in output.data() {
        assert!(val.is_finite(), "Output contains non-finite values");
    }
}

#[test]
fn test_transformer_block_residual_connection() {
    let block = TransformerBlock::new(4, 1, 16, 1e-5).expect("test");

    // Input must be 2D [seq_len, hidden_dim]
    // With zero input, output should be non-zero due to residual + processing
    let input = Tensor::from_vec(vec![1, 4], vec![0.0, 0.0, 0.0, 0.0]).expect("test");
    let output = block.forward(&input).expect("test");

    // Even with zero input, layer norm and attention should produce non-zero output
    // (though it might be small due to normalization)
    assert_eq!(output.shape(), &[1, 4]);
}

#[test]
fn test_transformer_block_shape_mismatch_error() {
    let block = TransformerBlock::new(8, 1, 32, 1e-5).expect("test");

    // Wrong hidden_dim (input has 4, block expects 8)
    let input = Tensor::from_vec(vec![4], vec![1.0; 4]).expect("test");
    let result = block.forward(&input);
    assert!(result.is_err());
}

#[test]
fn test_transformer_block_mutable_access() {
    let mut block = TransformerBlock::new(4, 1, 16, 1e-5).expect("test");

    // Verify we can access mutable references
    let _attn_norm = block.attn_norm_mut();
    let _attention = block.attention_mut();
    let _ffn_norm = block.ffn_norm_mut();
    let _ffn = block.ffn_mut();
}

// Embedding tests

#[test]
fn test_embedding_creation() {
    let embed = Embedding::new(1000, 64).expect("test");
    assert_eq!(embed.vocab_size(), 1000);
    assert_eq!(embed.embed_dim(), 64);
}

#[test]
fn test_embedding_zero_params_error() {
    assert!(Embedding::new(0, 64).is_err());
    assert!(Embedding::new(1000, 0).is_err());
}

#[test]
fn test_embedding_forward_shape() {
    let embed = Embedding::new(100, 8).expect("test");

    let token_ids = vec![0, 1, 2];
    let output = embed.forward(&token_ids).expect("test");

    assert_eq!(output.shape(), &[3, 8]);
    assert_eq!(output.data().len(), 24);
}

#[test]
fn test_embedding_forward_lookup() {
    let mut embed = Embedding::new(10, 4).expect("test");

    // Set specific embedding for token 5
    let offset = 5 * 4;
    embed.weights_mut()[offset] = 1.0;
    embed.weights_mut()[offset + 1] = 2.0;
    embed.weights_mut()[offset + 2] = 3.0;
    embed.weights_mut()[offset + 3] = 4.0;

    let output = embed.forward(&[5]).expect("test");
    assert_eq!(output.shape(), &[1, 4]);
    assert!((output.data()[0] - 1.0).abs() < 1e-6);
    assert!((output.data()[1] - 2.0).abs() < 1e-6);
    assert!((output.data()[2] - 3.0).abs() < 1e-6);
    assert!((output.data()[3] - 4.0).abs() < 1e-6);
}

#[test]
fn test_embedding_out_of_bounds_error() {
    let embed = Embedding::new(10, 4).expect("test");
    assert!(embed.forward(&[10]).is_err()); // ID 10 is out of bounds
    assert!(embed.forward(&[100]).is_err());
}

#[test]
fn test_embedding_empty_input_error() {
    let embed = Embedding::new(10, 4).expect("test");
    assert!(embed.forward(&[]).is_err());
}

// Model tests

#[test]
fn test_model_creation() {
    let config = ModelConfig {
        vocab_size: 100,
        hidden_dim: 8,
        num_heads: 1,
        num_layers: 2,
        intermediate_dim: 32,
        eps: 1e-5,
    };
    let model = Model::new(config).expect("test");
    assert_eq!(model.config().vocab_size, 100);
    assert_eq!(model.config().num_layers, 2);
}

#[test]
fn test_model_forward_shape() {
    let config = ModelConfig {
        vocab_size: 50,
        hidden_dim: 4,
        num_heads: 1,
        num_layers: 1,
        intermediate_dim: 16,
        eps: 1e-5,
    };
    let model = Model::new(config).expect("test");

    let token_ids = vec![0, 1, 2];
    let output = model.forward(&token_ids).expect("test");

    // Output should be [seq_len, vocab_size]
    assert_eq!(output.shape(), &[3, 50]);
}

#[test]
fn test_model_forward_valid_output() {
    let config = ModelConfig {
        vocab_size: 20,
        hidden_dim: 4,
        num_heads: 1,
        num_layers: 1,
        intermediate_dim: 16,
        eps: 1e-5,
    };
    let model = Model::new(config).expect("test");

    let output = model.forward(&[0, 1]).expect("test");

    // Output should be finite
    for &val in output.data() {
        assert!(val.is_finite(), "Output contains non-finite values");
    }
}

#[test]
fn test_model_num_parameters() {
    let config = ModelConfig {
        vocab_size: 100,
        hidden_dim: 8,
        num_heads: 1,
        num_layers: 2,
        intermediate_dim: 32,
        eps: 1e-5,
    };
    let model = Model::new(config).expect("test");

    let params = model.num_parameters();
    assert!(params > 0);
    // Should be at least embedding + lm_head
    assert!(params >= 100 * 8 + 8 * 100);
}

#[test]
fn test_model_mutable_access() {
    let config = ModelConfig {
        vocab_size: 50,
        hidden_dim: 4,
        num_heads: 1,
        num_layers: 1,
        intermediate_dim: 16,
        eps: 1e-5,
    };
    let mut model = Model::new(config).expect("test");

    // Verify we can access mutable references
    let _embed = model.embedding_mut();
    let _blocks = model.blocks_mut();
    let _norm = model.final_norm_mut();
    let _head = model.lm_head_mut();
}

#[test]
fn test_model_generate_basic() {
    let config = ModelConfig {
        vocab_size: 20,
        hidden_dim: 4,
        num_heads: 1,
        num_layers: 1,
        intermediate_dim: 16,
        eps: 1e-5,
    };
    let model = Model::new(config).expect("test");

    let gen_config = GenerationConfig::greedy().with_max_tokens(5);
    let tokens = model.generate(&[0], &gen_config).expect("test");

    // Should have prompt + up to 5 generated tokens
    assert!(tokens.len() <= 6);
    assert!(!tokens.is_empty());
    // First token should be prompt
    assert_eq!(tokens[0], 0);
}

include!("model_generate.rs");
include!("phase3_flash.rs");
include!("fused_qkv.rs");
