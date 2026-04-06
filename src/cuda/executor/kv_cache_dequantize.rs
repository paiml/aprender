
#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    // ========================================================================
    // QWEN-007 Phase 3: GPU Dequantization Tests
    // ========================================================================

    #[test]
    fn test_dequantize_kv_q8_gpu_not_enabled() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Without Q8 cache enabled
        let result = exec.dequantize_kv_q8_gpu(0, 1);
        assert!(result.is_err());
        match result {
            Err(e) => assert!(e.to_string().contains("not enabled")),
            Ok(_) => panic!("Expected error"),
        }
    }

    #[test]
    fn test_dequantize_kv_q8_gpu_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let num_kv_heads = 2;
        let head_dim = 64; // Divisible by 32
        let max_len = 8;

        // Initialize Q8 KV cache
        exec.init_kv_cache_q8_gpu(1, 4, num_kv_heads, head_dim, max_len)
            .expect("expected value");

        let size = num_kv_heads * head_dim;

        // Create test K/V vectors
        let k: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let v: Vec<f32> = (0..size).map(|i| -(i as f32) * 0.1).collect();

        // Write to position 0
        exec.write_kv_q8(0, 0, &k, &v).expect("write_kv_q8");

        // Dequantize on GPU
        let (k_fp32, v_fp32) = exec
            .dequantize_kv_q8_gpu(0, 1)
            .expect("GPU dequantization failed");

        // Download and verify
        let mut k_out = vec![0.0f32; size];
        let mut v_out = vec![0.0f32; size];
        k_fp32.copy_to_host(&mut k_out).expect("copy_to_host");
        v_fp32.copy_to_host(&mut v_out).expect("copy_to_host");

        // Verify values are close (Q8 has ~1% quantization error)
        for i in 0..size {
            let k_err = (k[i] - k_out[i]).abs();
            let v_err = (v[i] - v_out[i]).abs();
            let k_tol = (k[i].abs() * 0.02).max(0.02);
            let v_tol = (v[i].abs() * 0.02).max(0.02);
            assert!(
                k_err < k_tol,
                "K[{}]: expected {}, got {}, err {} > tol {}",
                i,
                k[i],
                k_out[i],
                k_err,
                k_tol
            );
            assert!(
                v_err < v_tol,
                "V[{}]: expected {}, got {}, err {} > tol {}",
                i,
                v[i],
                v_out[i],
                v_err,
                v_tol
            );
        }
    }

    #[test]
    fn test_dequantize_kv_q8_gpu_multiple_positions() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let num_kv_heads = 2;
        let head_dim = 32;
        let max_len = 16;
        let seq_len = 4;

        exec.init_kv_cache_q8_gpu(1, 4, num_kv_heads, head_dim, max_len)
            .expect("expected value");

        let size = num_kv_heads * head_dim;

        // Write to multiple positions
        for pos in 0..seq_len {
            let k: Vec<f32> = (0..size).map(|i| (pos as f32 + i as f32) * 0.05).collect();
            let v: Vec<f32> = (0..size).map(|i| -(pos as f32 + i as f32) * 0.05).collect();
            exec.write_kv_q8(0, pos, &k, &v).expect("write_kv_q8");
        }

        // Dequantize all positions on GPU
        let (k_fp32, v_fp32) = exec
            .dequantize_kv_q8_gpu(0, seq_len)
            .expect("GPU dequantization failed");

        // Verify buffer sizes
        let expected_size = seq_len * num_kv_heads * head_dim;
        let mut k_out = vec![0.0f32; expected_size];
        let mut v_out = vec![0.0f32; expected_size];
        k_fp32.copy_to_host(&mut k_out).expect("copy_to_host");
        v_fp32.copy_to_host(&mut v_out).expect("copy_to_host");

        assert_eq!(k_out.len(), expected_size);
        assert_eq!(v_out.len(), expected_size);
    }

    #[test]
    fn test_dequantize_kv_q8_gpu_vs_cpu() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let num_kv_heads = 2;
        let head_dim = 64;
        let max_len = 8;
        let seq_len = 3;

        exec.init_kv_cache_q8_gpu(1, 4, num_kv_heads, head_dim, max_len)
            .expect("expected value");

        let size = num_kv_heads * head_dim;

        // Write test data
        for pos in 0..seq_len {
            let k: Vec<f32> = (0..size)
                .map(|i| ((pos * size + i) as f32) * 0.01)
                .collect();
            let v: Vec<f32> = (0..size)
                .map(|i| -((pos * size + i) as f32) * 0.01)
                .collect();
            exec.write_kv_q8(0, pos, &k, &v).expect("write_kv_q8");
        }

        // CPU dequantization (read_kv_q8) - layout: [seq_len, num_kv_heads, head_dim]
        let (k_cpu, v_cpu) = exec.read_kv_q8(0, 0, seq_len).expect("read_kv_q8");

        // GPU dequantization - layout: [num_kv_heads, seq_len, head_dim]
        let (k_gpu_buf, v_gpu_buf) = exec.dequantize_kv_q8_gpu(0, seq_len).expect("dequantize_kv_q8_gpu");
        let mut k_gpu = vec![0.0f32; seq_len * size];
        let mut v_gpu = vec![0.0f32; seq_len * size];
        k_gpu_buf.copy_to_host(&mut k_gpu).expect("copy_to_host");
        v_gpu_buf.copy_to_host(&mut v_gpu).expect("copy_to_host");

        // Compare with layout transformation:
        // CPU index: pos * (num_kv_heads * head_dim) + head * head_dim + d
        // GPU index: head * (seq_len * head_dim) + pos * head_dim + d
        for pos in 0..seq_len {
            for head in 0..num_kv_heads {
                for d in 0..head_dim {
                    let cpu_idx = pos * (num_kv_heads * head_dim) + head * head_dim + d;
                    let gpu_idx = head * (seq_len * head_dim) + pos * head_dim + d;

                    let k_diff = (k_cpu[cpu_idx] - k_gpu[gpu_idx]).abs();
                    let v_diff = (v_cpu[cpu_idx] - v_gpu[gpu_idx]).abs();

                    // Allow tiny floating-point differences
                    assert!(
                        k_diff < 1e-6,
                        "K[pos={}, head={}, d={}] CPU={} GPU={} diff={}",
                        pos,
                        head,
                        d,
                        k_cpu[cpu_idx],
                        k_gpu[gpu_idx],
                        k_diff
                    );
                    assert!(
                        v_diff < 1e-6,
                        "V[pos={}, head={}, d={}] CPU={} GPU={} diff={}",
                        pos,
                        head,
                        d,
                        v_cpu[cpu_idx],
                        v_gpu[gpu_idx],
                        v_diff
                    );
                }
            }
        }
    }

    #[test]
    fn test_dequantize_kv_q8_gpu_exceeds_max_len() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        exec.init_kv_cache_q8_gpu(1, 4, 2, 32, 8).expect("init_kv_cache_q8_gpu");

        // Request more than max_len
        let result = exec.dequantize_kv_q8_gpu(0, 16);
        assert!(result.is_err());
        match result {
            Err(e) => assert!(e.to_string().contains("exceeds max_len")),
            Ok(_) => panic!("Expected error"),
        }
    }

    // ========================================================================
    // QWEN-007 Phase 4: Q8 Incremental Attention Tests
    // ========================================================================

    #[test]
    fn test_incremental_attention_q8_gpu_not_enabled() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Without Q8 cache enabled
        let q = vec![1.0f32; 256];
        let k = vec![1.0f32; 64];
        let v = vec![1.0f32; 64];
        let mut output = vec![0.0f32; 256];

        let result = exec.incremental_attention_q8_gpu(0, &q, &k, &v, &mut output);
        assert!(result.is_err());
        match result {
            Err(e) => assert!(e.to_string().contains("not enabled")),
            Ok(_) => panic!("Expected error"),
        }
    }

    #[test]
    fn test_incremental_attention_q8_gpu_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 64;
        let max_len = 16;

        // Initialize Q8 KV cache
        exec.init_kv_cache_q8_gpu(1, num_heads, num_kv_heads, head_dim, max_len)
            .expect("Q8 KV cache init failed");

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Create test vectors
        let q: Vec<f32> = (0..q_dim).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> = (0..kv_dim).map(|i| (i as f32) * 0.01).collect();
        let v: Vec<f32> = (0..kv_dim).map(|i| (i as f32) * 0.01).collect();
        let mut output = vec![0.0f32; q_dim];

        // Run Q8 attention
        let result = exec.incremental_attention_q8_gpu(0, &q, &k, &v, &mut output);
        assert!(result.is_ok(), "Q8 attention failed: {:?}", result.err());
        assert_eq!(result.expect("result"), 1, "Should return seq_len=1");

        // Output should be non-zero
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() > 0.0, "Output should be non-zero");
    }

    #[test]
    fn test_incremental_attention_q8_gpu_multiple_tokens() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 32; // Smaller for faster test
        let max_len = 8;

        exec.init_kv_cache_q8_gpu(1, num_heads, num_kv_heads, head_dim, max_len)
            .expect("Q8 KV cache init failed");

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Process multiple tokens
        for token_idx in 0..4 {
            let q: Vec<f32> = (0..q_dim)
                .map(|i| ((token_idx + i) as f32) * 0.01)
                .collect();
            let k: Vec<f32> = (0..kv_dim)
                .map(|i| ((token_idx + i) as f32) * 0.01)
                .collect();
            let v: Vec<f32> = (0..kv_dim)
                .map(|i| ((token_idx + i) as f32) * 0.01)
                .collect();
            let mut output = vec![0.0f32; q_dim];

            let result = exec.incremental_attention_q8_gpu(0, &q, &k, &v, &mut output);
            assert!(
                result.is_ok(),
                "Token {} Q8 attention failed: {:?}",
                token_idx,
                result.err()
            );
            assert_eq!(
                result.expect("result"),
                token_idx + 1,
                "Should return seq_len={}",
                token_idx + 1
            );
        }

        // Verify cache length
        assert_eq!(exec.kv_cache_len(0), 4);
    }

    #[test]
    fn test_incremental_attention_q8_gpu_dimension_mismatch() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 32;
        let max_len = 8;

        exec.init_kv_cache_q8_gpu(1, num_heads, num_kv_heads, head_dim, max_len)
            .expect("Q8 KV cache init failed");

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Wrong Q dimension
        let q_wrong = vec![1.0f32; q_dim + 10];
        let k = vec![1.0f32; kv_dim];
        let v = vec![1.0f32; kv_dim];
        let mut output = vec![0.0f32; q_dim];

        let result = exec.incremental_attention_q8_gpu(0, &q_wrong, &k, &v, &mut output);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("dimension mismatch"));
    }

    #[test]
    fn test_incremental_attention_q8_gpu_overflow() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 32;
        let max_len = 4; // Small for quick overflow test

        exec.init_kv_cache_q8_gpu(1, num_heads, num_kv_heads, head_dim, max_len)
            .expect("Q8 KV cache init failed");

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q = vec![1.0f32; q_dim];
        let k = vec![1.0f32; kv_dim];
        let v = vec![1.0f32; kv_dim];
        let mut output = vec![0.0f32; q_dim];

        // Fill cache to max
        for _ in 0..max_len {
            let result = exec.incremental_attention_q8_gpu(0, &q, &k, &v, &mut output);
            assert!(result.is_ok());
        }

        // Next should overflow
        let result = exec.incremental_attention_q8_gpu(0, &q, &k, &v, &mut output);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("overflow"));
    }
include!("kv_cache_init_tests.rs");
}
