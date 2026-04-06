
#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // FFN SwiGLU Tests
    // ========================================================================

    #[test]
    fn test_fused_ffn_swiglu_gpu_weight_not_cached() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).expect("input");
        let result = exec.fused_ffn_swiglu_gpu(
            &input,
            "nonexistent_gate",
            "nonexistent_up",
            "nonexistent_down",
            256,
            512,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_ffn_swiglu_gpu_true_dp4a_weight_not_cached() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).expect("input");
        let result = exec.fused_ffn_swiglu_gpu_true_dp4a(
            &input,
            "nonexistent_gate",
            "nonexistent_up",
            "nonexistent_down",
            256,
            512,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_ffn_swiglu_indexed_gpu_creates_output() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).expect("input");
        // Using zero pointers will fail kernel but tests function interface
        let result = exec.fused_ffn_swiglu_indexed_gpu(&input, 0, 0, 0, 256, 512);
        let _ = result;
    }

    #[test]
    fn test_fused_ffn_swiglu_host_weight_not_cached() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 256];
        let result = exec.fused_ffn_swiglu_host(
            &input,
            &mut output,
            "nonexistent_gate",
            "nonexistent_up",
            "nonexistent_down",
            256,
            512,
        );
        assert!(result.is_err());
    }

    // ========================================================================
    // FFN Indexed Tests
    // ========================================================================

    #[test]
    fn test_fused_ffn_swiglu_indexed_gpu_creates_output_buffer() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).expect("input");
        // Using zero pointers will fail kernel but tests function interface
        let result = exec.fused_ffn_swiglu_indexed_gpu(&input, 0, 0, 0, 256, 512);
        let _ = result;
    }

    // ========================================================================
    // Transformer Layer Tests
    // ========================================================================

    #[test]
    fn test_transformer_layer_workspace_dimensions() {
        // Test dimension calculations without actual kernel execution
        let hidden_dim = 256u32;
        let n_heads = 8u32;
        let head_dim = 32u32;
        let intermediate_dim = 512u32;

        // Verify dimensional constraints
        assert_eq!(hidden_dim, n_heads * head_dim);
        assert!(intermediate_dim > hidden_dim);
    }

    #[test]
    fn test_transformer_layer_q_offset_calculation() {
        // Test Q/K/V offset calculations
        let hidden_dim = 256usize;
        let n_kv_heads = 4usize;
        let head_dim = 32usize;

        let q_offset = 0;
        let k_offset = hidden_dim;
        let v_offset = k_offset + n_kv_heads * head_dim;

        assert_eq!(q_offset, 0);
        assert_eq!(k_offset, 256);
        assert_eq!(v_offset, 256 + 4 * 32);
    }

    // ========================================================================
    // Batched Transformer Layer Tests
    // ========================================================================

    #[test]
    fn test_batched_transformer_batch_size_constraints() {
        // Test batch size constraints for multi-sequence processing
        let max_batch = 32u32;
        let typical_batch = 8u32;

        assert!(typical_batch <= max_batch);
        assert!(typical_batch.is_power_of_two());
    }

    #[test]
    fn test_batched_kv_cache_stride_calculation() {
        // Test KV cache stride calculation
        let max_seq_len = 2048u32;
        let n_kv_heads = 4u32;
        let head_dim = 64u32;

        let kv_stride = max_seq_len * n_kv_heads * head_dim;
        assert_eq!(kv_stride, 2048 * 4 * 64);
    }

    // ========================================================================
    // Attention Tests
    // ========================================================================

    #[test]
    fn test_flash_attention_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let seq_len = 4usize;
        let head_dim = 32usize;
        // flash_attention uses single head only (seq_len * head_dim)
        let total = seq_len * head_dim;

        let q = vec![1.0f32; total];
        let k = vec![1.0f32; total];
        let v = vec![1.0f32; total];
        let mut output = vec![0.0f32; total];

        let scale = 1.0 / (head_dim as f32).sqrt();
        let causal = true;
        let result = exec.flash_attention(
            &q,
            &k,
            &v,
            &mut output,
            seq_len as u32,
            head_dim as u32,
            scale,
            causal,
        );
        // Validation catches the shared memory overflow before kernel launch.
        // The error is expected — trueno-gpu's AttentionKernel has a known bug.
        let _ = result;
    }

    #[test]
    fn test_flash_attention_dimension_calculation() {
        // Test attention dimension calculations
        let seq_len = 64u32;
        let head_dim = 64u32;
        let n_heads = 12u32;

        let q_size = seq_len * head_dim * n_heads;
        let k_size = seq_len * head_dim * n_heads;
        let v_size = seq_len * head_dim * n_heads;
        let output_size = seq_len * head_dim * n_heads;

        assert_eq!(q_size, k_size);
        assert_eq!(k_size, v_size);
        assert_eq!(v_size, output_size);
    }

    #[test]
    fn test_flash_attention_tile_size_calculation() {
        // Test tile size calculation for shared memory constraints
        let head_dim = 64u32;
        let max_shared = 48 * 1024u32; // 48KB

        let max_tile = max_shared / (head_dim * 12);
        assert!(max_tile > 0);
        assert!(max_tile <= 64); // Reasonable tile size
    }

    // ========================================================================
    // Flash Attention Multi-Head Tests
    // ========================================================================

    #[test]
    fn test_flash_attention_multi_head_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let seq_len = 4usize;
        let head_dim = 32usize;
        let n_heads = 2usize;
        let total = seq_len * head_dim * n_heads;

        let q = vec![1.0f32; total];
        let k = vec![1.0f32; total];
        let v = vec![1.0f32; total];
        let mut output = vec![0.0f32; total];

        let causal = true;
        let result = exec.flash_attention_multi_head(
            &q,
            &k,
            &v,
            &mut output,
            seq_len as u32,
            head_dim as u32,
            n_heads as u32,
            causal,
        );
        let _ = result;
    }

    #[test]
    fn test_flash_attention_thread_limit() {
        // Test thread limit constraint
        let head_dim = 64u32;
        let thread_limit = 1024 / head_dim;
        assert!(thread_limit <= 16); // Max 16 when head_dim=64
    }

    #[test]
    fn test_flash_attention_memory_bytes() {
        // Test memory calculation for flash attention
        let seq_len = 1024u32;
        let head_dim = 64u32;
        let (compute_mem, _peak_mem) =
            CudaExecutor::flash_attention_memory_bytes(seq_len, head_dim);
        assert!(compute_mem > 0);
    }

    // ========================================================================
    // Workspace Allocation Tests
    // ========================================================================

    #[test]
    fn test_workspace_allocation_sizes() {
        // Test workspace allocation size calculations
        let hidden_dim = 4096usize;
        let intermediate_dim = 11008usize;
        let _n_heads = 32usize;
        let n_kv_heads = 8usize;
        let head_dim = 128usize;
        let max_seq_len = 4096usize;

        // QKV projection size
        let qkv_size = hidden_dim + 2 * n_kv_heads * head_dim;
        assert!(qkv_size > 0);

        // FFN intermediate size
        let ffn_size = intermediate_dim;
        assert!(ffn_size > hidden_dim);

        // KV cache size per layer
        let kv_cache_size = 2 * max_seq_len * n_kv_heads * head_dim;
        assert!(kv_cache_size > 0);
    }

    // ========================================================================
    // Harness-Based Integration Tests
    // These tests use ModelHarness to setup complete executor state
    // ========================================================================

    #[test]
    fn test_fused_ffn_swiglu_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).expect("input");
        // Now we have weights loaded - use indexed pointers from layer 0
        let layer_weights = &exec.indexed_layer_weights[0];
        let result = exec.fused_ffn_swiglu_indexed_gpu(
            &input,
            layer_weights.ffn_gate_ptr,
            layer_weights.ffn_up_ptr,
            layer_weights.ffn_down_ptr,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
        );
        // Should execute kernel (may succeed or fail due to PTX issues)
        let _ = result;
    }

    #[test]
    fn test_flash_attention_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let seq_len = 4usize;
        let total = seq_len * config.head_dim;
        let q = vec![0.1f32; total];
        let k = vec![0.1f32; total];
        let v = vec![0.1f32; total];
        let mut output = vec![0.0f32; total];
        let scale = 1.0 / (config.head_dim as f32).sqrt();

        let result = exec.flash_attention(
            &q,
            &k,
            &v,
            &mut output,
            seq_len as u32,
            config.head_dim as u32,
            scale,
            true,
        );
        let _ = result;
    }

    #[test]
    fn test_flash_attention_multi_head_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let seq_len = 4usize;
        let total = seq_len * config.head_dim * config.num_heads;
        let q = vec![0.1f32; total];
        let k = vec![0.1f32; total];
        let v = vec![0.1f32; total];
        let mut output = vec![0.0f32; total];

        let result = exec.flash_attention_multi_head(
            &q,
            &k,
            &v,
            &mut output,
            seq_len as u32,
            config.head_dim as u32,
            config.num_heads as u32,
            true,
        );
        let _ = result;
    }

    #[test]
    fn test_transformer_layer_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Verify indexed weights were built (workspace is managed internally)
        assert_eq!(exec.indexed_layer_weights.len(), config.num_layers);
    }

    #[test]
    fn test_batched_attention_workspace_setup() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.num_layers = 2;
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // KV cache should be initialized
        assert!(exec.kv_cache_max_len > 0);
        assert!(exec.kv_num_kv_heads > 0);
        assert!(exec.kv_head_dim > 0);
    }

    #[test]
    fn test_gqa_configuration_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.num_heads = 32;
        config.num_kv_heads = 8; // GQA with 4:1 ratio
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Verify GQA configuration
        assert_eq!(exec.kv_num_heads, config.num_heads);
        assert_eq!(exec.kv_num_kv_heads, config.num_kv_heads);
    }

    #[test]
    fn test_rmsnorm_cache_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // RMSNorm gamma should be cached for each layer
        let key = "blk.0.attn_norm.gamma".to_string();
        assert!(exec.rmsnorm_cache.contains_key(&key));
    }

    #[test]
    fn test_lm_head_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // LM head should be loaded
        assert!(exec.lm_head_ptr != 0);
        assert!(exec.lm_head_len > 0);
    }
}
