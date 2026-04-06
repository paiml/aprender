
/// PAR-044: Pre-allocated workspace buffers for transformer forward pass
///
/// Eliminates ~288 GPU buffer allocations per token by reusing pre-sized buffers.
/// All buffers are allocated once at model load and reused for every token.
///
/// Performance impact:
/// - Before: ~288 cuMemAlloc calls per token (~2-3ms overhead)
/// - After: 0 allocations per token (all reused)
#[derive(Default)]
pub struct TransformerWorkspace {
    /// Hidden state buffer 1 (hidden_dim) - for normed, projected, ffn_normed, ffn_down
    pub hidden_buf1: Option<GpuBuffer<f32>>,
    /// Hidden state buffer 2 (hidden_dim) - for residual1, output
    pub hidden_buf2: Option<GpuBuffer<f32>>,
    /// Input staging buffer (hidden_dim) - preserves input for residual connections
    pub input_staging: Option<GpuBuffer<f32>>,
    /// Q/attention output buffer (q_dim)
    pub q_buf: Option<GpuBuffer<f32>>,
    /// K projection buffer (kv_dim)
    pub k_buf: Option<GpuBuffer<f32>>,
    /// V projection buffer (kv_dim)
    pub v_buf: Option<GpuBuffer<f32>>,
    /// FFN gate buffer (intermediate_dim)
    pub ffn_gate_buf: Option<GpuBuffer<f32>>,
    /// FFN up buffer (intermediate_dim)
    pub ffn_up_buf: Option<GpuBuffer<f32>>,
    /// FFN activated buffer (intermediate_dim) - result of SwiGLU
    pub ffn_act_buf: Option<GpuBuffer<f32>>,
    /// Attention output buffer (q_dim) - result of incremental attention
    /// PAR-051: Eliminates 28 GPU allocations per token
    pub attn_out_buf: Option<GpuBuffer<f32>>,
    /// PAR-054: Logits output buffer (vocab_size) - for CUDA graph capture
    pub logits_buf: Option<GpuBuffer<f32>>,
    /// PAR-054: Normed hidden buffer (hidden_dim) - for CUDA graph capture
    pub normed_hidden_buf: Option<GpuBuffer<f32>>,
    /// Workspace is initialized
    pub initialized: bool,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Q dimension (num_heads × head_dim)
    pub q_dim: usize,
    /// KV dimension (num_kv_heads × head_dim)
    pub kv_dim: usize,
    /// Intermediate dimension (FFN)
    pub intermediate_dim: usize,
    /// PAR-111: Batch size for multi-sequence processing (default 1)
    pub batch_size: usize,
    /// PMAT-045: High-water mark of allocated buffer capacity.
    /// Tracks the largest batch_size ever allocated — never shrinks.
    /// Skip conditions check this instead of batch_size to prevent
    /// workspace thrashing when prefill (large) and decode (small) alternate.
    pub buffer_capacity: usize,
    /// PAR-114: Positions buffer for batched RoPE (M positions)
    pub positions_buf: Option<GpuBuffer<u32>>,
    /// PAR-PERF-DP4A: Pre-allocated Q8_1 activation buffer for DP4A GEMV
    /// Eliminates per-GEMV cudaMalloc (was 280 mallocs/token → 0)
    /// Size: ceil(hidden_dim / 32) × 36 bytes (Q8_1 format)
    pub q8_activation_buf: Option<GpuBuffer<u8>>,
}

// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create zeroed `IndexedLayerWeights` for tests.
    /// PMAT-232: `Default` was intentionally removed to enforce explicit
    /// construction from GGUF metadata in production code.
    fn test_zeroed_layer_weights() -> IndexedLayerWeights {
        IndexedLayerWeights {
            attn_q_ptr: 0,
            attn_q_len: 0,
            attn_q_qtype: WeightQuantType::Q4K,
            attn_k_ptr: 0,
            attn_k_len: 0,
            attn_k_qtype: WeightQuantType::Q4K,
            attn_v_ptr: 0,
            attn_v_len: 0,
            attn_v_qtype: WeightQuantType::Q4K,
            attn_output_ptr: 0,
            attn_output_len: 0,
            attn_output_qtype: WeightQuantType::Q4K,
            ffn_gate_ptr: 0,
            ffn_gate_len: 0,
            ffn_gate_qtype: WeightQuantType::Q4K,
            ffn_up_ptr: 0,
            ffn_up_len: 0,
            ffn_up_qtype: WeightQuantType::Q4K,
            ffn_down_ptr: 0,
            ffn_down_len: 0,
            ffn_down_qtype: WeightQuantType::Q4K,
            attn_norm_ptr: 0,
            attn_norm_len: 0,
            ffn_norm_ptr: 0,
            ffn_norm_len: 0,
            attn_q_bias_ptr: 0,
            attn_q_bias_len: 0,
            attn_k_bias_ptr: 0,
            attn_k_bias_len: 0,
            attn_v_bias_ptr: 0,
            attn_v_bias_len: 0,
            attn_q_norm_ptr: 0,
            attn_q_norm_len: 0,
            attn_k_norm_ptr: 0,
            attn_k_norm_len: 0,
        }
    }

    // =========================================================================
    // WeightQuantType Tests
    // =========================================================================

    #[test]
    fn test_weight_quant_type_no_default() {
        // PMAT-232: WeightQuantType must NOT have a Default impl.
        // Every construction must be explicit to prevent silent wrong-kernel dispatch.
        // If this test fails to compile, the contract is enforced correctly.
        let explicit = WeightQuantType::Q4K;
        assert_eq!(explicit, WeightQuantType::Q4K);
    }

    #[test]
    fn test_weight_quant_type_bytes_per_superblock() {
        assert_eq!(WeightQuantType::Q4K.bytes_per_superblock(), 144);
        assert_eq!(WeightQuantType::Q5K.bytes_per_superblock(), 176);
        assert_eq!(WeightQuantType::Q6K.bytes_per_superblock(), 210);
        assert_eq!(WeightQuantType::Q8_0.bytes_per_superblock(), 34 * 8); // 272
        assert_eq!(WeightQuantType::Q5_0.bytes_per_superblock(), 22 * 8); // 176
        assert_eq!(WeightQuantType::Q4_0.bytes_per_superblock(), 18 * 8); // 144
        assert_eq!(WeightQuantType::Q4_1.bytes_per_superblock(), 20 * 8); // 160
    }

    #[test]
    fn test_weight_quant_type_bytes_per_block() {
        assert_eq!(WeightQuantType::Q4K.bytes_per_block(), 18);
        assert_eq!(WeightQuantType::Q5K.bytes_per_block(), 22);
        assert_eq!(WeightQuantType::Q6K.bytes_per_block(), 26);
        assert_eq!(WeightQuantType::Q8_0.bytes_per_block(), 34);
        assert_eq!(WeightQuantType::Q5_0.bytes_per_block(), 22);
        assert_eq!(WeightQuantType::Q4_0.bytes_per_block(), 18);
        assert_eq!(WeightQuantType::Q4_1.bytes_per_block(), 20);
    }

    #[test]
    fn test_weight_quant_type_from_ggml_type() {
        assert_eq!(
            WeightQuantType::from_ggml_type(2),
            Some(WeightQuantType::Q4_0)
        );
        assert_eq!(
            WeightQuantType::from_ggml_type(3),
            Some(WeightQuantType::Q4_1)
        );
        assert_eq!(
            WeightQuantType::from_ggml_type(6),
            Some(WeightQuantType::Q5_0)
        );
        assert_eq!(
            WeightQuantType::from_ggml_type(8),
            Some(WeightQuantType::Q8_0)
        );
        assert_eq!(
            WeightQuantType::from_ggml_type(12),
            Some(WeightQuantType::Q4K)
        );
        assert_eq!(
            WeightQuantType::from_ggml_type(13),
            Some(WeightQuantType::Q5K)
        );
        assert_eq!(
            WeightQuantType::from_ggml_type(14),
            Some(WeightQuantType::Q6K)
        );
        assert_eq!(WeightQuantType::from_ggml_type(99), None);
        assert_eq!(WeightQuantType::from_ggml_type(0), None);
    }

    #[test]
    fn test_weight_quant_type_matches_size_superblock() {
        // Q4K: 144 bytes per 256 elements
        // For 1024 rows × 256 cols: 1024 super-blocks × 144 = 147456 bytes
        assert!(WeightQuantType::Q4K.matches_size(147_456, 1024, 256));
        assert!(!WeightQuantType::Q4K.matches_size(147_457, 1024, 256)); // Wrong size

        // Q5K: 176 bytes per 256 elements
        assert!(WeightQuantType::Q5K.matches_size(1024 * 176, 1024, 256));

        // Q6K: 210 bytes per 256 elements
        assert!(WeightQuantType::Q6K.matches_size(1024 * 210, 1024, 256));
    }

    #[test]
    fn test_weight_quant_type_matches_size_block() {
        // Q4_0: 18 bytes per 32 elements
        // For 1024 rows × 32 cols: 1024 blocks × 18 = 18432 bytes
        assert!(WeightQuantType::Q4_0.matches_size(18_432, 1024, 32));

        // Q8_0: 34 bytes per 32 elements
        assert!(WeightQuantType::Q8_0.matches_size(1024 * 34, 1024, 32));

        // Q5_0: 22 bytes per 32 elements
        assert!(WeightQuantType::Q5_0.matches_size(1024 * 22, 1024, 32));

        // Q4_1: 20 bytes per 32 elements
        assert!(WeightQuantType::Q4_1.matches_size(1024 * 20, 1024, 32));
    }

    #[test]
    fn test_weight_quant_type_matches_size_partial_blocks() {
        // Test with non-aligned dimensions
        // 1024 rows × 100 cols for Q4_0: (100 + 31) / 32 = 4 blocks per row
        // 1024 × 4 × 18 = 73728 bytes
        assert!(WeightQuantType::Q4_0.matches_size(73_728, 1024, 100));
    }

    #[test]
    fn test_weight_quant_type_from_size_superblock() {
        // Q4K: 144 bytes per 256 elements
        let size = 1024 * 144; // 1024 super-blocks
        assert_eq!(
            WeightQuantType::from_size(size, 1024, 256),
            Some(WeightQuantType::Q4K)
        );

        // Q5K: 176 bytes per 256 elements
        let size = 512 * 176;
        assert_eq!(
            WeightQuantType::from_size(size, 512, 256),
            Some(WeightQuantType::Q5K)
        );

        // Q6K: 210 bytes per 256 elements
        let size = 256 * 210;
        assert_eq!(
            WeightQuantType::from_size(size, 256, 256),
            Some(WeightQuantType::Q6K)
        );
    }

    #[test]
    fn test_weight_quant_type_from_size_block() {
        // Q4_0: 18 bytes per 32 elements
        let size = 1024 * 18;
        assert_eq!(
            WeightQuantType::from_size(size, 1024, 32),
            Some(WeightQuantType::Q4_0)
        );

        // Q8_0: 34 bytes per 32 elements
        let size = 512 * 34;
        assert_eq!(
            WeightQuantType::from_size(size, 512, 32),
            Some(WeightQuantType::Q8_0)
        );

        // Q5_0: 22 bytes per 32 elements
        let size = 256 * 22;
        assert_eq!(
            WeightQuantType::from_size(size, 256, 32),
            Some(WeightQuantType::Q5_0)
        );

        // Q4_1: 20 bytes per 32 elements
        let size = 128 * 20;
        assert_eq!(
            WeightQuantType::from_size(size, 128, 32),
            Some(WeightQuantType::Q4_1)
        );
    }

    #[test]
    fn test_weight_quant_type_from_size_none() {
        // Size that doesn't match any format
        assert_eq!(WeightQuantType::from_size(12345, 100, 256), None);
    }

    #[test]
    fn test_weight_quant_type_clone_eq() {
        let qtype = WeightQuantType::Q6K;
        let cloned = qtype;
        assert_eq!(qtype, cloned);
    }

    #[test]
    fn test_weight_quant_type_debug() {
        let qtype = WeightQuantType::Q4K;
        let debug = format!("{:?}", qtype);
        assert!(debug.contains("Q4K"));
    }

    // =========================================================================
    // IndexedLayerWeights Tests
    // =========================================================================

    #[test]
    fn test_indexed_layer_weights_zeroed() {
        let weights = test_zeroed_layer_weights();
        assert_eq!(weights.attn_q_ptr, 0);
        assert_eq!(weights.attn_q_len, 0);
        assert_eq!(weights.attn_q_qtype, WeightQuantType::Q4K);
        assert_eq!(weights.ffn_gate_ptr, 0);
        assert_eq!(weights.attn_norm_len, 0);
    }

    #[test]
    fn test_indexed_layer_weights_clone() {
        let mut weights = test_zeroed_layer_weights();
        weights.attn_q_ptr = 12345;
        weights.attn_q_len = 1024;
        weights.attn_q_qtype = WeightQuantType::Q5K;

        let cloned = weights.clone();
        assert_eq!(cloned.attn_q_ptr, 12345);
        assert_eq!(cloned.attn_q_len, 1024);
        assert_eq!(cloned.attn_q_qtype, WeightQuantType::Q5K);
    }

    #[test]
    fn test_indexed_layer_weights_debug() {
        let weights = test_zeroed_layer_weights();
        let debug = format!("{:?}", weights);
        assert!(debug.contains("IndexedLayerWeights"));
    }

    #[test]
    fn test_indexed_layer_weights_all_fields() {
        let weights = IndexedLayerWeights {
            attn_q_ptr: 100,
            attn_q_len: 1024,
            attn_q_qtype: WeightQuantType::Q4K,
            attn_k_ptr: 200,
            attn_k_len: 512,
            attn_k_qtype: WeightQuantType::Q5K,
            attn_v_ptr: 300,
            attn_v_len: 512,
            attn_v_qtype: WeightQuantType::Q6K,
            attn_output_ptr: 400,
            attn_output_len: 1024,
            attn_output_qtype: WeightQuantType::Q4_0,
            ffn_gate_ptr: 500,
            ffn_gate_len: 4096,
            ffn_gate_qtype: WeightQuantType::Q4K,
            ffn_up_ptr: 600,
            ffn_up_len: 4096,
            ffn_up_qtype: WeightQuantType::Q4K,
            ffn_down_ptr: 700,
            ffn_down_len: 1024,
            ffn_down_qtype: WeightQuantType::Q6K,
            attn_norm_ptr: 800,
            attn_norm_len: 1024,
            ffn_norm_ptr: 900,
            ffn_norm_len: 1024,
            attn_q_bias_ptr: 0,
            attn_q_bias_len: 0,
            attn_k_bias_ptr: 0,
            attn_k_bias_len: 0,
            attn_v_bias_ptr: 0,
            attn_v_bias_len: 0,
            attn_q_norm_ptr: 0,
            attn_q_norm_len: 0,
            attn_k_norm_ptr: 0,
            attn_k_norm_len: 0,
        };

        assert_eq!(weights.attn_q_ptr, 100);
        assert_eq!(weights.ffn_down_qtype, WeightQuantType::Q6K);
        assert_eq!(weights.attn_q_bias_len, 0); // No bias
    }

    // =========================================================================
    // TransformerWorkspace Tests
    // =========================================================================

    #[test]
    fn test_transformer_workspace_default() {
        let workspace = TransformerWorkspace::default();
        assert!(!workspace.initialized);
        assert_eq!(workspace.hidden_dim, 0);
        assert_eq!(workspace.q_dim, 0);
        assert_eq!(workspace.kv_dim, 0);
        assert_eq!(workspace.intermediate_dim, 0);
        assert_eq!(workspace.batch_size, 0);
        assert_eq!(workspace.buffer_capacity, 0);
        assert!(workspace.hidden_buf1.is_none());
        assert!(workspace.hidden_buf2.is_none());
        assert!(workspace.q_buf.is_none());
        assert!(workspace.k_buf.is_none());
        assert!(workspace.v_buf.is_none());
        assert!(workspace.logits_buf.is_none());
    }
}
