
#[test]
fn test_kv_cache_head_dim_cov() {
    let cache = KVCache::new(4, 512, 128).expect("cache");
    assert_eq!(cache.head_dim(), 128);
}

#[test]
fn test_kv_cache_current_pos_cov() {
    let cache = KVCache::new(2, 512, 64).expect("cache");
    assert_eq!(cache.current_pos(), 0);
}

// =========================================================================
// Coverage Tests: RoPE (Rotary Position Embeddings)
// =========================================================================

#[test]
fn test_rope_zero_dim_cov() {
    let result = RoPE::new(0, 10000.0);
    assert!(result.is_err());
}

#[test]
fn test_rope_odd_dim_cov() {
    let result = RoPE::new(63, 10000.0);
    assert!(result.is_err());
}

#[test]
fn test_rope_small_base_cov() {
    // RoPE with small base should still work (no validation on base)
    let result = RoPE::new(64, 1.0);
    assert!(result.is_ok());
}

#[test]
fn test_rope_dim_accessor_cov() {
    let rope = RoPE::new(64, 10000.0).expect("rope");
    assert_eq!(rope.dim(), 64);
}

#[test]
fn test_rope_base_accessor_cov() {
    let rope = RoPE::new(64, 10000.0).expect("rope");
    assert!((rope.base() - 10000.0).abs() < 1e-6);
}

// =========================================================================
// Coverage Tests: ScaledRoPE
// =========================================================================

#[test]
fn test_scaled_rope_dim_accessor_cov() {
    let scaled = ScaledRoPE::new(64, 10000.0, RopeScalingType::None).expect("scaled");
    assert_eq!(scaled.dim(), 64);
}

#[test]
fn test_scaled_rope_linear_scaling_cov() {
    let scaling = RopeScalingType::Linear { scale: 2.0 };
    let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("scaled");
    assert_eq!(scaled.dim(), 64);
}

#[test]
fn test_scaled_rope_ntk_scaling_cov() {
    let scaling = RopeScalingType::Ntk { scale: 1.5 };
    let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("scaled");
    assert_eq!(scaled.dim(), 64);
}

// =========================================================================
// Coverage Tests: ALiBi
// =========================================================================

#[test]
fn test_alibi_zero_heads_cov() {
    let result = ALiBi::new(0);
    assert!(result.is_err());
}

#[test]
fn test_alibi_num_heads_accessor_cov() {
    let alibi = ALiBi::new(8).expect("alibi");
    assert_eq!(alibi.num_heads(), 8);
}

#[test]
fn test_alibi_power_of_two_cov() {
    // 16 is power of 2
    let alibi = ALiBi::new(16).expect("alibi");
    assert_eq!(alibi.num_heads(), 16);
}

#[test]
fn test_alibi_non_power_of_two_cov() {
    // 12 is not power of 2
    let alibi = ALiBi::new(12).expect("alibi");
    assert_eq!(alibi.num_heads(), 12);
}

// =========================================================================
// Coverage Tests: MultiHeadAttention
// =========================================================================

#[test]
fn test_mha_zero_hidden_dim_cov() {
    let result = MultiHeadAttention::new(0, 4, 4);
    assert!(result.is_err());
}

#[test]
fn test_mha_zero_heads_cov() {
    let result = MultiHeadAttention::new(256, 0, 0);
    assert!(result.is_err());
}

#[test]
fn test_mha_indivisible_dim_cov() {
    // 256 not divisible by 5
    let result = MultiHeadAttention::new(256, 5, 5);
    assert!(result.is_err());
}

#[test]
fn test_mha_num_heads_accessor_cov() {
    let mha = MultiHeadAttention::new(256, 4, 4).expect("mha");
    assert_eq!(mha.num_heads(), 4);
}

#[test]
fn test_mha_head_dim_accessor_cov() {
    let mha = MultiHeadAttention::new(256, 4, 4).expect("mha");
    assert_eq!(mha.head_dim(), 64); // 256 / 4 = 64
}

#[test]
fn test_mha_hidden_dim_accessor_cov() {
    let mha = MultiHeadAttention::new(512, 8, 8).expect("mha");
    assert_eq!(mha.hidden_dim(), 512);
}

// =========================================================================
// Coverage Tests: Embedding
// =========================================================================

#[test]
fn test_embedding_zero_vocab_cov() {
    let result = Embedding::new(0, 256);
    assert!(result.is_err());
}

#[test]
fn test_embedding_zero_dim_cov() {
    let result = Embedding::new(1000, 0);
    assert!(result.is_err());
}

#[test]
fn test_embedding_vocab_size_accessor_cov() {
    let emb = Embedding::new(50000, 768).expect("embedding");
    assert_eq!(emb.vocab_size(), 50000);
}

#[test]
fn test_embedding_embed_dim_accessor_cov() {
    let emb = Embedding::new(50000, 768).expect("embedding");
    assert_eq!(emb.embed_dim(), 768);
}

#[test]
fn test_embedding_weights_mut_cov() {
    let mut emb = Embedding::new(100, 64).expect("embedding");
    let weight = emb.weights_mut();
    assert_eq!(weight.len(), 100 * 64);
    weight[0] = 1.0;
    assert!((emb.weights_mut()[0] - 1.0).abs() < 1e-6);
}

// =========================================================================
// Coverage Tests: TransformerBlock
// =========================================================================

#[test]
fn test_transformer_block_hidden_dim_cov() {
    let block = TransformerBlock::new(256, 4, 1024, 1e-5).expect("block");
    assert_eq!(block.hidden_dim(), 256);
}

// =========================================================================
// Coverage Tests: SlidingWindowAttention
// =========================================================================

#[test]
fn test_sliding_window_zero_head_dim_cov() {
    let result = SlidingWindowAttention::new(0, 1024);
    assert!(result.is_err());
}

#[test]
fn test_sliding_window_zero_window_cov() {
    let result = SlidingWindowAttention::new(64, 0);
    assert!(result.is_err());
}

#[test]
fn test_sliding_window_window_size_accessor_cov() {
    let swa = SlidingWindowAttention::new(64, 2048).expect("swa");
    assert_eq!(swa.window_size(), 2048);
}

#[test]
fn test_sliding_window_head_dim_accessor_cov() {
    let swa = SlidingWindowAttention::new(128, 1024).expect("swa");
    assert_eq!(swa.head_dim(), 128);
}

// =========================================================================
// Coverage Tests: FusedQKVAttention
// =========================================================================

#[test]
fn test_fused_qkv_zero_head_dim_cov() {
    let result = FusedQKVAttention::new(0, 256);
    assert!(result.is_err());
}

#[test]
fn test_fused_qkv_indivisible_cov() {
    // 256 not divisible by 7
    let result = FusedQKVAttention::new(7, 256);
    assert!(result.is_err());
}

#[test]
fn test_fused_qkv_num_heads_accessor_cov() {
    let fused = FusedQKVAttention::new(64, 256).expect("fused");
    assert_eq!(fused.num_heads(), 4); // 256 / 64 = 4
}

#[test]
fn test_fused_qkv_head_dim_accessor_cov() {
    let fused = FusedQKVAttention::new(64, 256).expect("fused");
    assert_eq!(fused.head_dim(), 64);
}

#[test]
fn test_fused_qkv_hidden_dim_accessor_cov() {
    let fused = FusedQKVAttention::new(64, 512).expect("fused");
    assert_eq!(fused.hidden_dim(), 512);
}

// =========================================================================
// Coverage Tests: Model
// =========================================================================

#[test]
fn test_model_config_accessor_cov() {
    let config = ModelConfig {
        vocab_size: 32000,
        hidden_dim: 512,
        num_layers: 6,
        num_heads: 8,
        intermediate_dim: 2048,
        eps: 1e-5,
    };
    let model = Model::new(config.clone()).expect("model");
    assert_eq!(model.config().vocab_size, 32000);
    assert_eq!(model.config().num_layers, 6);
}
