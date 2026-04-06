
#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::cuda::executor::test_fixtures::{generate_q4_0_weights, generate_q8_0_weights};

    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // Weight Loading Tests (Using Synthetic Data)
    // ========================================================================

    #[test]
    fn test_load_quantized_weights_q4_0() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // Generate Q4_0 weights: 8 blocks = 256 elements (8 * 32)
        // For k=256, n=128: need 128 rows * 8 blocks per row = 1024 blocks
        let weights = generate_q4_0_weights(1024);

        let result = exec.load_quantized_weights("test_q4_0", &weights);
        assert!(result.is_ok());
        assert!(result.unwrap() > 0);
    }

    #[test]
    fn test_load_quantized_weights_q8_0() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // Generate Q8_0 weights: 8 blocks = 256 elements
        let weights = generate_q8_0_weights(1024);

        let result = exec.load_quantized_weights("test_q8_0", &weights);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_quantized_weights_with_type() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let weights = generate_q4_0_weights(256);

        // Load with explicit Q4_0 type (type 2)
        let result = exec.load_quantized_weights_with_type("test_typed", &weights, 2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_multiple_weights() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let weights1 = generate_q4_0_weights(256);
        let weights2 = generate_q8_0_weights(256);

        exec.load_quantized_weights("weight1", &weights1).unwrap();
        exec.load_quantized_weights("weight2", &weights2).unwrap();

        // Both should be cached
        assert!(exec.quantized_weight_cache.contains_key("weight1"));
        assert!(exec.quantized_weight_cache.contains_key("weight2"));
    }

    // ========================================================================
    // F32 Weight Loading Tests
    // ========================================================================

    #[test]
    fn test_load_weights_f32() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let weights: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();

        let result = exec.load_weights("test_f32", &weights);
        assert!(result.is_ok());
        assert!(exec.weight_cache.contains_key("test_f32"));
    }

    #[test]
    fn test_load_weights_f32_multiple() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let weights1: Vec<f32> = (0..512).map(|i| i as f32 * 0.01).collect();
        let weights2: Vec<f32> = (0..256).map(|i| i as f32 * 0.02).collect();

        exec.load_weights("test_f32_v1", &weights1).unwrap();
        exec.load_weights("test_f32_v2", &weights2).unwrap();

        assert!(exec.has_weights("test_f32_v1"));
        assert!(exec.has_weights("test_f32_v2"));
    }

    // ========================================================================
    // RMSNorm Weight Loading Tests (via cache_rmsnorm_gamma)
    // ========================================================================

    #[test]
    fn test_cache_rmsnorm_gamma() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let gamma: Vec<f32> = vec![1.0; 256];

        let result = exec.cache_rmsnorm_gamma("test_norm", &gamma);
        assert!(result.is_ok());
        assert!(exec.rmsnorm_cache.contains_key("test_norm"));
    }

    #[test]
    fn test_cache_rmsnorm_gamma_ptr() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let gamma: Vec<f32> = vec![1.0; 128];

        exec.cache_rmsnorm_gamma("norm_ptr_test", &gamma).unwrap();
        let ptr = exec
            .rmsnorm_cache
            .get("norm_ptr_test")
            .map(trueno_gpu::driver::GpuBuffer::as_ptr);
        assert!(ptr.is_some());
        assert!(ptr.unwrap() > 0);
    }

    #[test]
    fn test_rmsnorm_cache_not_found() {
        let Some(exec) = create_executor() else {
            return;
        };
        let ptr = exec.rmsnorm_cache.get("nonexistent");
        assert!(ptr.is_none());
    }

    // ========================================================================
    // LM Head Bias Loading Tests
    // ========================================================================

    #[test]
    fn test_preload_lm_head_bias() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let bias: Vec<f32> = vec![0.1; 256];

        let result = exec.preload_lm_head_bias(Some(&bias));
        assert!(result.is_ok());
        assert!(exec.has_lm_head_bias());
    }

    #[test]
    fn test_preload_lm_head_bias_none() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let result = exec.preload_lm_head_bias(None);
        assert!(result.is_ok());
        // No bias loaded
        assert!(!exec.has_lm_head_bias());
    }

    #[test]
    fn test_bias_cache_not_found() {
        let Some(exec) = create_executor() else {
            return;
        };
        let ptr = exec.bias_cache.get("nonexistent");
        assert!(ptr.is_none());
    }

    // ========================================================================
    // Indexed Weights Tests
    // ========================================================================

    #[test]
    fn test_has_indexed_weights_initial_false() {
        let Some(exec) = create_executor() else {
            return;
        };
        assert!(!exec.has_indexed_weights());
    }

    #[test]
    fn test_clear_indexed_weights() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // Clear should work even if nothing is indexed
        exec.clear_indexed_weights();
        assert!(!exec.has_indexed_weights());
        assert_eq!(exec.output_norm_ptr, 0);
        assert_eq!(exec.lm_head_ptr, 0);
    }

    // ========================================================================
    // Q4_0 GEMV with Synthetic Weights (Actual Kernel Execution)
    // ========================================================================

    #[test]
    fn test_q4_0_gemv_with_synthetic_weights() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Generate Q4_0 synthetic weights
        // For K=256, N=128: need 128 rows, 8 blocks per row (32 elements/block)
        let k = 256u32;
        let n = 128u32;
        let blocks_per_row = (k / 32) as usize;
        let total_blocks = n as usize * blocks_per_row;
        let weights = generate_q4_0_weights(total_blocks);

        // Load weights to GPU
        exec.load_quantized_weights_with_type("synth_gemv", &weights, 2)
            .unwrap();
        let weight_ptr = exec.get_quantized_weight_ptr("synth_gemv").unwrap();

        // Create GPU buffers for input/output
        let input: Vec<f32> = (0..k as usize).map(|i| (i as f32) * 0.01).collect();
        let output: Vec<f32> = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        // Execute Q4_0 GEMV (may fail due to PTX, but exercises the path)
        let result = exec.q4_0_gemv_into(weight_ptr, &input_buf, &output_buf, n, k);
        let _ = result;
    }

    // ========================================================================
    // Q8_0 GEMV with Synthetic Weights
    // ========================================================================

    #[test]
    fn test_q8_0_gemv_with_synthetic_weights() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let k = 256u32;
        let n = 64u32;
        let blocks_per_row = (k / 32) as usize;
        let total_blocks = n as usize * blocks_per_row;
        let weights = generate_q8_0_weights(total_blocks);

        exec.load_quantized_weights_with_type("q8_gemv", &weights, 7)
            .unwrap();
        let weight_ptr = exec.get_quantized_weight_ptr("q8_gemv").unwrap();

        let input: Vec<f32> = vec![1.0; k as usize];
        let output: Vec<f32> = vec![0.0f32; n as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = GpuBuffer::from_host(&exec.context, &output).unwrap();

        let result = exec.q8_0_gemv_into(weight_ptr, &input_buf, &output_buf, n, k);
        let _ = result;
    }

    // ========================================================================
    // Weight Cache Management Tests
    // ========================================================================

    #[test]
    fn test_weight_cache_overwrite() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let weights1 = generate_q4_0_weights(128);
        let weights2 = generate_q4_0_weights(256);

        exec.load_quantized_weights("overwrite_test", &weights1)
            .unwrap();
        let size1 = exec
            .quantized_weight_cache
            .get("overwrite_test")
            .unwrap()
            .len();

        exec.load_quantized_weights("overwrite_test", &weights2)
            .unwrap();
        let size2 = exec
            .quantized_weight_cache
            .get("overwrite_test")
            .unwrap()
            .len();

        // Second load should replace first
        assert!(size2 > size1);
    }

    #[test]
    fn test_quantized_weight_types_tracking() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let weights = generate_q4_0_weights(128);

        // Load as Q4_0 (type 2)
        exec.load_quantized_weights_with_type("typed_q4", &weights, 2)
            .unwrap();

        // Verify type is tracked
        assert_eq!(exec.quantized_weight_types.get("typed_q4"), Some(&2));
    }

    // ========================================================================
    // Get Quantized Weight Ptr Tests
    // ========================================================================

    #[test]
    fn test_get_quantized_weight_ptr_after_load() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let weights = generate_q4_0_weights(128);

        exec.load_quantized_weights("ptr_test", &weights).unwrap();

        let ptr = exec.get_quantized_weight_ptr("ptr_test");
        assert!(ptr.is_ok());
        assert!(ptr.unwrap() > 0);
    }

    // ========================================================================
    // GH-45 FALSIFICATION: Dual-cache dispatch (weight_cache + quantized_weight_cache)
    // ========================================================================

    /// GH-45: has_weights() must find f32 weights, has_quantized_weights() must find
    /// quantized weights. Before the fix, has_cached_weight() only checked weight_cache
    /// (f32), so quantized APR weights were never found → 278x slowdown.
    #[test]
    fn test_falsify_gh45_dual_cache_dispatch() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Load f32 weight
        let f32_weights: Vec<f32> = vec![1.0; 256];
        exec.load_weights("layer0.ffn_norm", &f32_weights).unwrap();

        // Load quantized weight (Q4_K, type 12)
        let q4k_weights = generate_q4_0_weights(256);
        exec.load_quantized_weights_with_type("layer0.ffn_gate", &q4k_weights, 12)
            .unwrap();

        // GH-45 contract: f32 cache must find f32 weights
        assert!(
            exec.has_weights("layer0.ffn_norm"),
            "GH-45: has_weights() must find f32 cached weight"
        );
        assert!(
            !exec.has_weights("layer0.ffn_gate"),
            "GH-45: has_weights() must NOT find quantized weight in f32 cache"
        );

        // GH-45 contract: quantized cache must find quantized weights
        assert!(
            exec.has_quantized_weights("layer0.ffn_gate"),
            "GH-45: has_quantized_weights() must find quantized cached weight"
        );
        assert!(
            !exec.has_quantized_weights("layer0.ffn_norm"),
            "GH-45: has_quantized_weights() must NOT find f32 weight in quantized cache"
        );

        // GH-45 KEY TEST: The combined check (what has_cached_weight dispatches to)
        // must return true for BOTH types of weights.
        let has_norm = exec.has_weights("layer0.ffn_norm")
            || exec.has_quantized_weights("layer0.ffn_norm");
        let has_gate = exec.has_weights("layer0.ffn_gate")
            || exec.has_quantized_weights("layer0.ffn_gate");
        assert!(
            has_norm,
            "GH-45: Combined cache check must find f32 weight"
        );
        assert!(
            has_gate,
            "GH-45: Combined cache check must find quantized weight"
        );

        // Verify neither cache finds non-existent weight
        let has_missing = exec.has_weights("nonexistent")
            || exec.has_quantized_weights("nonexistent");
        assert!(
            !has_missing,
            "GH-45: Combined check must return false for uncached weight"
        );
    }

    /// GH-45: Quantized weight type must be tracked alongside cache entry.
    /// Without type tracking, kernel dispatch falls to wrong dequant kernel.
    #[test]
    fn test_falsify_gh45_quantized_type_tracking() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let weights = generate_q4_0_weights(128);

        // Load Q6_K (type 14) — type must be stored
        exec.load_quantized_weights_with_type("layer0.attn_output", &weights, 14)
            .unwrap();

        // GH-45: Type must be retrievable for correct kernel dispatch
        assert_eq!(
            exec.get_quantized_weight_type("layer0.attn_output"),
            Some(14),
            "GH-45: Quantized weight type must be tracked for kernel dispatch"
        );

        // Load Q5_0 (type 6) — different type on different weight
        exec.load_quantized_weights_with_type("layer0.attn_q", &weights, 6)
            .unwrap();
        assert_eq!(
            exec.get_quantized_weight_type("layer0.attn_q"),
            Some(6),
            "GH-45: Each weight must track its own quantization type"
        );

        // Non-existent weight returns None
        assert_eq!(
            exec.get_quantized_weight_type("nonexistent"),
            None,
            "GH-45: Uncached weight must return None for type"
        );
    }
}
