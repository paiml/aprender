
// ============================================================================
// Tests (T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Attention Tests
    // =========================================================================

    #[test]
    fn test_attention_new() {
        let attn = Attention::new(64).expect("valid head_dim");
        assert_eq!(attn.head_dim, 64);
        // Scale should be 1/sqrt(64) = 1/8 = 0.125
        assert!((attn.scale - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_attention_new_zero_head_dim() {
        let result = Attention::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_scale_calculation() {
        // head_dim=16: scale = 1/sqrt(16) = 1/4 = 0.25
        let attn = Attention::new(16).expect("valid");
        assert!((attn.scale - 0.25).abs() < 1e-6);

        // head_dim=64: scale = 1/sqrt(64) = 1/8 = 0.125
        let attn = Attention::new(64).expect("valid");
        assert!((attn.scale - 0.125).abs() < 1e-6);

        // head_dim=128: scale = 1/sqrt(128) ≈ 0.0884
        let attn = Attention::new(128).expect("valid");
        let expected = 1.0 / (128.0_f32).sqrt();
        assert!((attn.scale - expected).abs() < 1e-6);
    }

    #[test]
    fn test_attention_forward_simple() {
        let attn = Attention::new(4).expect("valid");

        // Create simple 2x4 tensors (seq_len=2, head_dim=4)
        let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("valid");
        let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("valid");
        let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("valid");

        let result = attn.forward(&q, &k, &v);
        assert!(result.is_ok());

        let output = result.expect("should succeed");
        assert_eq!(output.shape(), &[2, 4]);
    }

    #[test]
    fn test_attention_forward_shape_mismatch() {
        let attn = Attention::new(4).expect("valid");

        let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("valid");
        let k = Tensor::from_vec(vec![2, 8], vec![1.0; 16]).expect("valid"); // Wrong dim
        let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("valid");

        let result = attn.forward(&q, &k, &v);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_forward_1d_tensor() {
        let attn = Attention::new(4).expect("valid");

        // 1D tensor works (treated as seq_len=1)
        let q = Tensor::from_vec(vec![4], vec![1.0; 4]).expect("valid");
        let k = Tensor::from_vec(vec![4], vec![1.0; 4]).expect("valid");
        let v = Tensor::from_vec(vec![4], vec![1.0; 4]).expect("valid");

        let result = attn.forward(&q, &k, &v);
        assert!(result.is_ok());
    }

    // =========================================================================
    // SlidingWindowAttention Tests
    // =========================================================================

    #[test]
    fn test_sliding_window_attention_new() {
        let attn = SlidingWindowAttention::new(64, 512).expect("valid");
        assert_eq!(attn.head_dim(), 64);
        assert_eq!(attn.window_size(), 512);
    }

    #[test]
    fn test_sliding_window_attention_zero_head_dim() {
        let result = SlidingWindowAttention::new(0, 512);
        assert!(result.is_err());
    }

    #[test]
    fn test_sliding_window_attention_zero_window() {
        let result = SlidingWindowAttention::new(64, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_sliding_window_attention_forward() {
        let attn = SlidingWindowAttention::new(4, 8).expect("valid");

        let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("valid");
        let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("valid");
        let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("valid");

        let result = attn.forward(&q, &k, &v);
        assert!(result.is_ok());

        let output = result.expect("should succeed");
        assert_eq!(output.shape(), &[2, 4]);
    }

    // =========================================================================
    // FusedQKVAttention Tests
    // =========================================================================

    #[test]
    fn test_fused_qkv_attention_new() {
        // FusedQKVAttention::new(head_dim, hidden_dim)
        // hidden_dim must be divisible by head_dim
        let attn = FusedQKVAttention::new(64, 256).expect("valid");
        assert_eq!(attn.head_dim(), 64);
    }

    #[test]
    fn test_fused_qkv_attention_zero_head_dim() {
        let result = FusedQKVAttention::new(0, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_qkv_attention_zero_hidden_dim() {
        let result = FusedQKVAttention::new(64, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_qkv_attention_hidden_not_divisible() {
        // hidden_dim=100 is not divisible by head_dim=64
        let result = FusedQKVAttention::new(64, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_qkv_attention_forward() {
        // FusedQKVAttention takes input [seq_len, hidden_dim] and projects Q, K, V internally
        let attn = FusedQKVAttention::new(4, 8).expect("valid"); // head_dim=4, hidden_dim=8

        // Input: [seq_len=2, hidden_dim=8]
        let input = Tensor::from_vec(vec![2, 8], vec![1.0; 16]).expect("valid");

        let result = attn.forward(&input);
        assert!(result.is_ok());

        let output = result.expect("should succeed");
        assert_eq!(output.shape(), &[2, 8]); // Output: [seq_len, hidden_dim]
    }

    // =========================================================================
    // MultiHeadAttention Tests
    // =========================================================================

    #[test]
    fn test_mha_new() {
        // MultiHeadAttention::new(hidden_dim, num_heads, num_kv_heads)
        // head_dim = hidden_dim / num_heads = 256 / 4 = 64
        let mha = MultiHeadAttention::new(256, 4, 4).expect("valid");
        assert_eq!(mha.num_heads(), 4);
        assert_eq!(mha.num_kv_heads(), 4);
        assert_eq!(mha.head_dim(), 64);
        assert_eq!(mha.hidden_dim(), 256);
    }

    #[test]
    fn test_mha_gqa() {
        // GQA: 8 query heads, 2 KV heads
        // head_dim = 512 / 8 = 64
        let mha = MultiHeadAttention::new(512, 8, 2).expect("valid");
        assert!(mha.is_gqa());
        assert!(!mha.is_mqa());
        assert!(!mha.is_mha());
    }

    #[test]
    fn test_mha_mqa() {
        // MQA: 8 query heads, 1 KV head
        let mha = MultiHeadAttention::new(512, 8, 1).expect("valid");
        assert!(mha.is_mqa());
        assert!(!mha.is_gqa());
        assert!(!mha.is_mha());
    }

    #[test]
    fn test_mha_standard() {
        // Standard MHA: num_heads == num_kv_heads
        let mha = MultiHeadAttention::new(512, 8, 8).expect("valid");
        assert!(mha.is_mha());
        assert!(!mha.is_gqa());
        assert!(!mha.is_mqa());
    }

    #[test]
    fn test_mha_zero_hidden_dim() {
        let result = MultiHeadAttention::new(0, 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_mha_zero_heads() {
        let result = MultiHeadAttention::new(256, 0, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_mha_zero_kv_heads() {
        let result = MultiHeadAttention::new(256, 4, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_mha_hidden_dim_not_divisible() {
        // hidden_dim=101 is not divisible by num_heads=4
        let result = MultiHeadAttention::new(101, 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_mha_kv_heads_not_divisor() {
        // num_heads=8 must be divisible by num_kv_heads=3, but 8 % 3 != 0
        let result = MultiHeadAttention::new(512, 8, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_mha_kv_heads_greater_than_heads() {
        // num_kv_heads=16 > num_heads=8 is invalid
        let result = MultiHeadAttention::new(512, 8, 16);
        assert!(result.is_err());
    }
}
