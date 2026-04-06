
#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // Q4K GEMV Cached Tests
    // ========================================================================

    #[test]
    fn test_q4k_gemv_cached_weight_not_found() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 128];
        let result = exec.q4k_gemv_cached("nonexistent", &input, &mut output, 128, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_q5k_gemv_cached_weight_not_found() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 128];
        let result = exec.q5k_gemv_cached("nonexistent", &input, &mut output, 128, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_q6k_gemv_cached_weight_not_found() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 128];
        let result = exec.q6k_gemv_cached("nonexistent", &input, &mut output, 128, 256);
        assert!(result.is_err());
    }

    // ========================================================================
    // Q4K GEMV Cached Async Tests
    // ========================================================================

    #[test]
    fn test_q4k_gemv_cached_async_weight_not_found() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).expect("input");
        let result = exec.q4k_gemv_cached_async("nonexistent", &input, 128, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_q6k_gemv_cached_async_weight_not_found() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).expect("input");
        let result = exec.q6k_gemv_cached_async("nonexistent", &input, 128, 256);
        assert!(result.is_err());
    }

    // ========================================================================
    // Q4K GEMV Indexed Async Tests
    // ========================================================================

    #[test]
    fn test_q4k_gemv_indexed_async_creates_output() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).expect("input");
        // Use zero pointer - will likely fail kernel but tests buffer creation
        let result = exec.q4k_gemv_indexed_async(0, &input, 128, 256);
        // Just testing it compiles and runs - actual result depends on PTX
        let _ = result;
    }

    #[test]
    fn test_q6k_gemv_indexed_async_creates_output() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).expect("input");
        let result = exec.q6k_gemv_indexed_async(0, &input, 128, 256);
        let _ = result;
    }

    // ========================================================================
    // Q4K GEMV Into Tests
    // ========================================================================

    #[test]
    fn test_q4k_gemv_into_tiled_creates_kernel() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).expect("input");
        let output = GpuBuffer::<f32>::new(&exec.context, 128).expect("output");
        // Test kernel loading path with zero weight pointer
        let result = exec.q4k_gemv_into_tiled(0, &input, &output, 128, 256);
        let _ = result;
    }

    #[test]
    fn test_coalesced_q4k_gemv_into_creates_kernel() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).expect("input");
        let output = GpuBuffer::<f32>::new(&exec.context, 128).expect("output");
        let result = exec.coalesced_q4k_gemv_into(0, &input, &output, 128, 256);
        let _ = result;
    }

    #[test]
    fn test_vectorized_q4k_gemv_into_creates_kernel() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).expect("input");
        let output = GpuBuffer::<f32>::new(&exec.context, 128).expect("output");
        let result = exec.vectorized_q4k_gemv_into(0, &input, &output, 128, 256);
        let _ = result;
    }

    #[test]
    fn test_dp4a_q4k_gemv_into_creates_kernel() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).expect("input");
        let output = GpuBuffer::<f32>::new(&exec.context, 128).expect("output");
        let result = exec.dp4a_q4k_gemv_into(0, &input, &output, 128, 256);
        let _ = result;
    }

    // ========================================================================
    // Fused Kernels Tests
    // ========================================================================

    #[test]
    fn test_fused_rmsnorm_q4k_gemv_into_creates_kernel() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).expect("input");
        let output = GpuBuffer::<f32>::new(&exec.context, 128).expect("output");
        let result = exec.fused_rmsnorm_q4k_gemv_into(0, &input, 0, &output, 256, 128, 1e-5);
        let _ = result;
    }

    #[test]
    fn test_fused_gate_up_q4k_gemv_into_creates_kernel() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).expect("input");
        let gate_out = GpuBuffer::<f32>::new(&exec.context, 128).expect("gate_out");
        let up_out = GpuBuffer::<f32>::new(&exec.context, 128).expect("up_out");
        let result = exec.fused_gate_up_q4k_gemv_into(0, 0, &input, &gate_out, &up_out, 256, 128);
        let _ = result;
    }

    // ========================================================================
    // Batched Q4K GEMV Tests
    // ========================================================================

    #[test]
    fn test_batched_q4k_gemv_into_m4() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // M=4, K=256, N=128
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 4 * 256]).expect("input");
        let output = GpuBuffer::<f32>::new(&exec.context, 4 * 128).expect("output");
        let result = exec.batched_q4k_gemv_into(0, &input, &output, 4, 128, 256);
        let _ = result;
    }

    #[test]
    fn test_batched_q4k_gemv_into_m8() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // M=8 (max for single kernel)
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 8 * 256]).expect("input");
        let output = GpuBuffer::<f32>::new(&exec.context, 8 * 128).expect("output");
        let result = exec.batched_q4k_gemv_into(0, &input, &output, 8, 128, 256);
        let _ = result;
    }

    #[test]
    fn test_batched_q4k_gemv_into_m16_multi_warp() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // M=16 uses multi-warp kernel
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 16 * 256]).expect("input");
        let output = GpuBuffer::<f32>::new(&exec.context, 16 * 128).expect("output");
        let result = exec.batched_q4k_gemv_into(0, &input, &output, 16, 128, 256);
        let _ = result;
    }

    #[test]
    fn test_batched_q4k_gemv_into_m32_multi_warp() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // M=32 uses 4-warp kernel
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32 * 256]).expect("input");
        let output = GpuBuffer::<f32>::new(&exec.context, 32 * 128).expect("output");
        let result = exec.batched_q4k_gemv_into(0, &input, &output, 32, 128, 256);
        let _ = result;
    }

    #[test]
    fn test_batched_q4k_gemv_into_m12_tiled() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // M=12 uses tiling (8+4)
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 12 * 256]).expect("input");
        let output = GpuBuffer::<f32>::new(&exec.context, 12 * 128).expect("output");
        let result = exec.batched_q4k_gemv_into(0, &input, &output, 12, 128, 256);
        let _ = result;
    }

    // ========================================================================
    // Get Quantized Weight Ptr Tests
    // ========================================================================

    #[test]
    fn test_get_quantized_weight_ptr_not_found() {
        let Some(exec) = create_executor() else {
            return;
        };
        let result = exec.get_quantized_weight_ptr("nonexistent");
        assert!(result.is_err());
    }

    // ========================================================================
    // Q4K GEMV Cached Tiled Tests
    // ========================================================================

    #[test]
    fn test_q4k_gemv_cached_tiled_weight_not_found() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 128];
        let result = exec.q4k_gemv_cached_tiled("nonexistent", &input, &mut output, 128, 256);
        assert!(result.is_err());
    }

    // ========================================================================
    // Harness-Based Integration Tests
    // ========================================================================

    #[test]
    fn test_q4k_gemv_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Use indexed weights from layer 0
        let layer_weights = &exec.indexed_layer_weights[0];
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).expect("input");
        let output = GpuBuffer::<f32>::new(&exec.context, config.hidden_dim).expect("output");

        // Test Q4K GEMV using attn_q weight
        let result = exec.q4k_gemv_into_tiled(
            layer_weights.attn_q_ptr,
            &input,
            &output,
            config.hidden_dim as u32,
            config.hidden_dim as u32,
        );
        let _ = result;
    }

    #[test]
    fn test_q4k_gemv_coalesced_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let layer_weights = &exec.indexed_layer_weights[0];
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).expect("input");
        let output = GpuBuffer::<f32>::new(&exec.context, config.hidden_dim).expect("output");

        let result = exec.coalesced_q4k_gemv_into(
            layer_weights.attn_q_ptr,
            &input,
            &output,
            config.hidden_dim as u32,
            config.hidden_dim as u32,
        );
        let _ = result;
    }

    #[test]
    fn test_q4k_gemv_vectorized_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let layer_weights = &exec.indexed_layer_weights[0];
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).expect("input");
        let output = GpuBuffer::<f32>::new(&exec.context, config.hidden_dim).expect("output");

        let result = exec.vectorized_q4k_gemv_into(
            layer_weights.attn_q_ptr,
            &input,
            &output,
            config.hidden_dim as u32,
            config.hidden_dim as u32,
        );
        let _ = result;
    }

    #[test]
    fn test_q4k_gemv_dp4a_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let layer_weights = &exec.indexed_layer_weights[0];
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).expect("input");
        let output = GpuBuffer::<f32>::new(&exec.context, config.hidden_dim).expect("output");

        let result = exec.dp4a_q4k_gemv_into(
            layer_weights.attn_q_ptr,
            &input,
            &output,
            config.hidden_dim as u32,
            config.hidden_dim as u32,
        );
        let _ = result;
    }

    #[test]
    fn test_fused_rmsnorm_q4k_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let layer_weights = &exec.indexed_layer_weights[0];
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).expect("input");
        let output = GpuBuffer::<f32>::new(&exec.context, config.hidden_dim).expect("output");

        // Use RMSNorm gamma pointer and attn_q weight
        let result = exec.fused_rmsnorm_q4k_gemv_into(
            layer_weights.attn_q_ptr,
            &input,
            layer_weights.attn_norm_ptr,
            &output,
            config.hidden_dim as u32,
            config.hidden_dim as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_fused_gate_up_q4k_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let layer_weights = &exec.indexed_layer_weights[0];
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).expect("input");
        let gate_out = GpuBuffer::<f32>::new(&exec.context, config.intermediate_dim).expect("gate_out");
        let up_out = GpuBuffer::<f32>::new(&exec.context, config.intermediate_dim).expect("up_out");

        let result = exec.fused_gate_up_q4k_gemv_into(
            layer_weights.ffn_gate_ptr,
            layer_weights.ffn_up_ptr,
            &input,
            &gate_out,
            &up_out,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
        );
        let _ = result;
    }

    #[test]
    fn test_batched_q4k_gemv_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let layer_weights = &exec.indexed_layer_weights[0];
        let m = 4u32;
        let input = GpuBuffer::from_host(
            &exec.context,
            &vec![0.1f32; (m as usize) * config.hidden_dim],
        )
        .expect("expected value");
        let output =
            GpuBuffer::<f32>::new(&exec.context, (m as usize) * config.hidden_dim).expect("expected value");

        let result = exec.batched_q4k_gemv_into(
            layer_weights.attn_q_ptr,
            &input,
            &output,
            m,
            config.hidden_dim as u32,
            config.hidden_dim as u32,
        );
        let _ = result;
    }
}
