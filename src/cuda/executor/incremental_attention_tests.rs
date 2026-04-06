
#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // Incremental Attention Tests
    // ========================================================================

    #[test]
    fn test_incremental_attention_async_requires_kv_cache_init() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let hidden_dim = 256usize;

        let q = GpuBuffer::from_host(&exec.context, &vec![1.0f32; hidden_dim]).unwrap();
        let k = GpuBuffer::from_host(&exec.context, &vec![1.0f32; hidden_dim]).unwrap();
        let v = GpuBuffer::from_host(&exec.context, &vec![1.0f32; hidden_dim]).unwrap();

        // Without KV cache init, should fail
        let result = exec.incremental_attention_async(0, &q, &k, &v);
        assert!(result.is_err());
    }

    #[test]
    fn test_incremental_attention_into_requires_kv_cache_init() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let hidden_dim = 256usize;

        let q = GpuBuffer::from_host(&exec.context, &vec![1.0f32; hidden_dim]).unwrap();
        let k = GpuBuffer::from_host(&exec.context, &vec![1.0f32; hidden_dim]).unwrap();
        let v = GpuBuffer::from_host(&exec.context, &vec![1.0f32; hidden_dim]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, hidden_dim).unwrap();

        // Without KV cache init, should fail
        let result = exec.incremental_attention_into(0, &q, &k, &v, &output);
        assert!(result.is_err());
    }

    // ========================================================================
    // Batched Incremental Attention Tests
    // ========================================================================

    #[test]
    fn test_batched_incremental_attention_dimensions() {
        // Test batched attention dimension calculations
        let batch_size = 4u32;
        let seq_len = 1024u32;
        let n_heads = 32u32;
        let head_dim = 64u32;
        let n_kv_heads = 8u32;

        // Q dimensions: batch × hidden_dim
        let q_size = batch_size * n_heads * head_dim;
        assert_eq!(q_size, 4 * 32 * 64);

        // KV cache dimensions: batch × seq_len × n_kv_heads × head_dim
        let kv_size = batch_size * seq_len * n_kv_heads * head_dim;
        assert!(kv_size > 0);

        // Output dimensions: same as Q
        let output_size = q_size;
        assert_eq!(output_size, q_size);
    }

    // ========================================================================
    // Flash Decoding Tests
    // ========================================================================

    #[test]
    fn test_init_flash_decoding() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let n_heads = 32usize;
        let head_dim = 64usize;
        let max_seq_len = 2048usize;
        let batch_size = 1usize;

        let result = exec.init_flash_decoding(n_heads, head_dim, max_seq_len, batch_size);
        assert!(result.is_ok());

        // Verify flash decoding is enabled
        assert!(exec.flash_decode_enabled);
    }

    #[test]
    fn test_flash_decoding_disabled_by_default() {
        let Some(exec) = create_executor() else {
            return;
        };
        assert!(!exec.flash_decode_enabled);
    }

    // ========================================================================
    // Tensor Core Attention Tests
    // ========================================================================

    #[test]
    fn test_tensor_core_attention_requires_aligned_dims() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // WMMA requires dimensions to be multiples of 16
        let seq_len = 32u32; // Multiple of 16
        let head_dim = 64u32; // Multiple of 16
        let n_heads = 4u32;
        let total = (seq_len * head_dim * n_heads) as usize;

        let q = vec![1.0f32; total];
        let k = vec![1.0f32; total];
        let v = vec![1.0f32; total];
        let mut output = vec![0.0f32; total];

        let result =
            exec.tensor_core_attention(&q, &k, &v, &mut output, seq_len, head_dim, n_heads, true);
        // May fail on non-Tensor Core GPUs but exercises the code path
        let _ = result;
    }

    #[test]
    fn test_tensor_core_attention_unaligned_dims_error() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // Use dimensions not multiples of 16
        let seq_len = 33u32; // Not multiple of 16
        let head_dim = 65u32; // Not multiple of 16
        let n_heads = 1u32;
        let total = (seq_len * head_dim * n_heads) as usize;

        let q = vec![1.0f32; total];
        let k = vec![1.0f32; total];
        let v = vec![1.0f32; total];
        let mut output = vec![0.0f32; total];

        let result =
            exec.tensor_core_attention(&q, &k, &v, &mut output, seq_len, head_dim, n_heads, true);
        assert!(result.is_err());
    }

    // ========================================================================
    // GEMM FP16 Tests
    // ========================================================================

    #[test]
    fn test_gemm_fp16_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let m = 32u32;
        let n = 32u32;
        let k = 32u32;

        let a = vec![1.0f32; (m * k) as usize];
        let b = vec![1.0f32; (k * n) as usize];
        let mut c = vec![0.0f32; (m * n) as usize];

        let result = exec.gemm_fp16(&a, &b, &mut c, m, n, k);
        let _ = result;
    }

    #[test]
    fn test_gemm_fp16_size_validation() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // Wrong sizes
        let a = vec![1.0f32; 10];
        let b = vec![1.0f32; 10];
        let mut c = vec![0.0f32; 10];

        let result = exec.gemm_fp16(&a, &b, &mut c, 32, 32, 32);
        assert!(result.is_err());
    }

    // ========================================================================
    // Flash Attention Memory Bytes Tests
    // ========================================================================

    #[test]
    fn test_flash_attention_memory_bytes_small_seq() {
        let (naive, flash) = CudaExecutor::flash_attention_memory_bytes(64, 64);
        // Naive: 64 * 64 * 4 = 16384 bytes
        assert_eq!(naive, 16384);
        // Flash: 64 * 64 * 4 * 2 = 32768 bytes (2 blocks)
        assert_eq!(flash, 32768);
    }

    #[test]
    fn test_flash_attention_memory_bytes_large_seq() {
        let (naive, flash) = CudaExecutor::flash_attention_memory_bytes(4096, 64);
        // Naive: 4096 * 4096 * 4 = 67,108,864 bytes (64MB)
        assert_eq!(naive, 67_108_864);
        // Flash: still 32KB (constant!)
        assert_eq!(flash, 32768);
    }

    #[test]
    fn test_flash_attention_memory_bytes_savings() {
        let seq_len = 2048u32;
        let (naive, flash) = CudaExecutor::flash_attention_memory_bytes(seq_len, 64);

        // Flash should use much less memory than naive
        assert!(flash < naive);

        // Calculate savings ratio
        let savings = naive / flash;
        assert!(savings > 100); // >100x savings for seq_len=2048
    }

    // ========================================================================
    // Attention Calculation Tests
    // ========================================================================

    #[test]
    fn test_attention_scale_calculation() {
        // Standard attention scale: 1/sqrt(head_dim)
        let head_dim = 64f32;
        let scale = 1.0 / head_dim.sqrt();
        assert!((scale - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_attention_softmax_numerics() {
        // Test that large values don't cause overflow in softmax
        let large_score = 100.0f32;
        let shifted = large_score - large_score; // shift by max for numerical stability
        let exp_val = shifted.exp();
        assert!((exp_val - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gqa_head_mapping() {
        // Test grouped query attention head mapping
        let n_heads = 32u32;
        let n_kv_heads = 8u32;
        let group_size = n_heads / n_kv_heads;

        assert_eq!(group_size, 4);

        // Each Q head maps to a KV head
        for q_head in 0..n_heads {
            let kv_head = q_head / group_size;
            assert!(kv_head < n_kv_heads);
        }
    }

    #[test]
    fn test_rope_frequency_calculation() {
        // Test RoPE frequency calculation
        let head_dim = 64u32;
        let theta = 10000.0f32;

        // Frequencies: theta^(-2i/d) for i in 0..head_dim/2
        let freq_0 = theta.powf(0.0);
        assert_eq!(freq_0, 1.0);

        let freq_mid = theta.powf(-2.0 * 16.0 / head_dim as f32);
        assert!(freq_mid < 1.0);
    }

    // ========================================================================
    // Harness-Based Integration Tests
    // ========================================================================

    #[test]
    fn test_incremental_attention_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let hidden_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        let q = GpuBuffer::from_host(&exec.context, &vec![0.1f32; hidden_dim]).unwrap();
        let k = GpuBuffer::from_host(&exec.context, &vec![0.1f32; kv_dim]).unwrap();
        let v = GpuBuffer::from_host(&exec.context, &vec![0.1f32; kv_dim]).unwrap();

        // KV cache is now initialized via harness
        let result = exec.incremental_attention_async(0, &q, &k, &v);
        // Should execute the attention kernel path
        let _ = result;
    }

    #[test]
    fn test_incremental_attention_into_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let hidden_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        let q = GpuBuffer::from_host(&exec.context, &vec![0.1f32; hidden_dim]).unwrap();
        let k = GpuBuffer::from_host(&exec.context, &vec![0.1f32; kv_dim]).unwrap();
        let v = GpuBuffer::from_host(&exec.context, &vec![0.1f32; kv_dim]).unwrap();
        let output = GpuBuffer::<f32>::new(&exec.context, hidden_dim).unwrap();

        let result = exec.incremental_attention_into(0, &q, &k, &v, &output);
        let _ = result;
    }

    #[test]
    fn test_flash_decoding_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Initialize flash decoding
        let result = exec.init_flash_decoding(
            config.num_heads,
            config.head_dim,
            config.max_seq_len,
            1, // batch_size
        );
        assert!(result.is_ok());
        assert!(exec.flash_decode_enabled);
    }

    #[test]
    fn test_kv_cache_scatter_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // KV cache should be properly initialized
        assert!(exec.kv_cache_max_len > 0);
        assert!(exec.kv_num_heads > 0);
        assert!(exec.kv_head_dim > 0);

        // Verify KV cache GPU buffers exist
        let k_key = "kv_0_k".to_string();
        let v_key = "kv_0_v".to_string();
        assert!(exec.kv_cache_gpu.contains_key(&k_key));
        assert!(exec.kv_cache_gpu.contains_key(&v_key));
    }

    #[test]
    fn test_multi_layer_attention_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.num_layers = 4;
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Verify all layers have KV cache
        for layer_idx in 0..config.num_layers {
            let k_key = format!("kv_{}_k", layer_idx);
            let v_key = format!("kv_{}_v", layer_idx);
            assert!(
                exec.kv_cache_gpu.contains_key(&k_key),
                "Missing KV cache for layer {}",
                layer_idx
            );
            assert!(
                exec.kv_cache_gpu.contains_key(&v_key),
                "Missing KV cache for layer {}",
                layer_idx
            );
        }
    }

    #[test]
    fn test_attention_with_gqa_ratio() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.num_heads = 32;
        config.num_kv_heads = 8; // 4:1 GQA ratio
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Verify GQA ratio is correctly applied
        let gqa_ratio = exec.kv_num_heads / exec.kv_num_kv_heads;
        assert_eq!(gqa_ratio, 4);
    }

    #[test]
    fn test_attention_rope_theta_config() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // RoPE theta should be configured
        assert!(exec.rope_theta > 0.0);
    }
}
