//! EXTREME TDD: Synthetic Model Tests for 95% cuda.rs Coverage
//!
//! Strategy G.1: SyntheticModel - 1 layer, 1 head, hidden_dim=32
//! Strategy G.2: Direct Kernel Injection - rmsnorm_into, fused_ffn with random vectors
//! Strategy G.5: Ghost Loader - mock weight loading
//!
//! These tests falsify cuda.rs logic WITHOUT loading real model files.

#![cfg(feature = "cuda")]
#![allow(unused_variables)]

use realizar::cuda::CudaExecutor;
use trueno_gpu::driver::GpuBuffer;

/// Helper to read GPU buffer to host
fn read_gpu_buffer(buf: &GpuBuffer<f32>, len: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; len];
    buf.copy_to_host(&mut result).expect("copy to host");
    result
}

// ============================================================================
// Strategy G.2: Direct Kernel Injection - Test functions with real GPU buffers
// ============================================================================

#[test]
fn test_g2_rmsnorm_gpu_direct() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed (RTX 4090 expected): {:?}", e);
            return;
        },
    };

    let hidden_dim = 32u32;
    let input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.1).collect();
    let gamma: Vec<f32> = vec![1.0; hidden_dim as usize];

    let ctx = exec.context();
    let input_gpu = GpuBuffer::from_host(ctx, &input).expect("input buffer");
    let gamma_gpu = GpuBuffer::from_host(ctx, &gamma).expect("gamma buffer");

    let result = exec.rmsnorm_gpu(&input_gpu, &gamma_gpu, hidden_dim, 1e-5);
    assert!(result.is_ok(), "rmsnorm_gpu failed: {:?}", result.err());

    let output_gpu = result.unwrap();
    let output = read_gpu_buffer(&output_gpu, hidden_dim as usize);

    let sum: f32 = output.iter().sum();
    assert!(
        sum.abs() > 1e-6,
        "Output should not be all zeros, got sum={}",
        sum
    );
}

#[test]
fn test_g2_rmsnorm_into_direct() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let hidden_dim = 64u32;
    let input: Vec<f32> = (0..hidden_dim).map(|i| ((i as f32) - 32.0) * 0.1).collect();
    let gamma: Vec<f32> = vec![1.0; hidden_dim as usize];

    let ctx = exec.context();
    let input_gpu = GpuBuffer::from_host(ctx, &input).expect("input buffer");
    let gamma_gpu = GpuBuffer::from_host(ctx, &gamma).expect("gamma buffer");
    let output_gpu = GpuBuffer::<f32>::new(ctx, hidden_dim as usize).expect("output buffer");

    let result = exec.rmsnorm_into(&input_gpu, &gamma_gpu, &output_gpu, hidden_dim, 1e-5);
    assert!(result.is_ok(), "rmsnorm_into failed: {:?}", result.err());

    exec.synchronize().expect("sync");
    let output = read_gpu_buffer(&output_gpu, hidden_dim as usize);

    let input_l2: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
    let output_l2: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (input_l2 - output_l2).abs() > 0.001,
        "RMSNorm should change L2 norm: input={}, output={}",
        input_l2,
        output_l2
    );
}

#[test]
fn test_g2_vectorized_rmsnorm() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let hidden_dim = 256u32;
    let input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.01).collect();
    let gamma: Vec<f32> = vec![1.0; hidden_dim as usize];

    let ctx = exec.context();
    let input_gpu = GpuBuffer::from_host(ctx, &input).expect("input");
    let gamma_gpu = GpuBuffer::from_host(ctx, &gamma).expect("gamma");
    let output_gpu = GpuBuffer::<f32>::new(ctx, hidden_dim as usize).expect("output");

    let result = exec.rmsnorm_into(&input_gpu, &gamma_gpu, &output_gpu, hidden_dim, 1e-5);
    assert!(
        result.is_ok(),
        "VectorizedRmsNorm failed: {:?}",
        result.err()
    );

    exec.synchronize().expect("sync");
    let output = read_gpu_buffer(&output_gpu, hidden_dim as usize);

    assert!(
        output.iter().all(|x| x.is_finite()),
        "All outputs should be finite"
    );
}

#[test]
fn test_g2_softmax_inplace() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let dim = 64usize;
    let mut data: Vec<f32> = (0..dim).map(|i| (i as f32) - 32.0).collect();

    let result = exec.softmax(&mut data);
    assert!(result.is_ok(), "softmax failed: {:?}", result.err());

    let sum: f32 = data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "Softmax should sum to 1.0, got {}",
        sum
    );
    assert!(
        data.iter().all(|&x| x >= 0.0),
        "Softmax outputs should be non-negative"
    );
}

#[test]
fn test_g2_gemm_small() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let m = 4u32;
    let k = 8u32;
    let n = 4u32;

    let a: Vec<f32> = vec![1.0; (m * k) as usize];
    let b: Vec<f32> = vec![1.0; (k * n) as usize];
    let mut c: Vec<f32> = vec![0.0; (m * n) as usize];

    let result = exec.gemm(&a, &b, &mut c, m, n, k);
    assert!(result.is_ok(), "GEMM failed: {:?}", result.err());

    for (i, val) in c.iter().enumerate() {
        assert!(
            (val - k as f32).abs() < 0.5,
            "c[{}]: expected {}, got {}",
            i,
            k,
            val
        );
    }
}

#[test]
fn test_g2_gemm_tiled() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let m = 64u32;
    let k = 64u32;
    let n = 64u32;

    let a: Vec<f32> = vec![0.1; (m * k) as usize];
    let b: Vec<f32> = vec![0.1; (k * n) as usize];
    let mut c: Vec<f32> = vec![0.0; (m * n) as usize];

    let result = exec.gemm(&a, &b, &mut c, m, n, k);
    assert!(result.is_ok(), "Tiled GEMM failed: {:?}", result.err());

    let expected = k as f32 * 0.01;
    for (i, val) in c.iter().enumerate() {
        assert!(
            (val - expected).abs() < 0.1,
            "c[{}]: expected ~{}, got {}",
            i,
            expected,
            val
        );
    }
}

// ============================================================================
// Strategy G.1: Synthetic Model - Workspace and KV Cache
// ============================================================================

#[test]
fn test_g1_synthetic_workspace_init() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let hidden_dim = 64usize;
    let intermediate_dim = 128usize;

    let result = exec.init_workspace(hidden_dim, intermediate_dim);
    assert!(result.is_ok(), "init_workspace failed: {:?}", result.err());
}

#[test]
fn test_g1_synthetic_kv_cache() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let n_layers = 2usize;
    let max_seq_len = 64usize;
    let n_heads = 2usize;
    let n_kv_heads = 2usize;
    let head_dim = 32usize;

    let result = exec.init_kv_cache_gpu(n_layers, max_seq_len, n_heads, n_kv_heads, head_dim);
    assert!(
        result.is_ok(),
        "init_kv_cache_gpu failed: {:?}",
        result.err()
    );
}

#[test]
fn test_g1_synthetic_rope_into() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let head_dim = 32u32;
    let n_heads = 2u32;
    let size = (head_dim * n_heads) as usize;

    let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

    let ctx = exec.context();
    let input_gpu = GpuBuffer::from_host(ctx, &input).expect("input");
    let output_gpu = GpuBuffer::<f32>::new(ctx, size).expect("output");

    let position = 5u32;
    let theta = 10000.0f32;

    // rope_into(input, output, position, num_heads, head_dim, theta)
    let result = exec.rope_into(&input_gpu, &output_gpu, position, n_heads, head_dim, theta);
    assert!(result.is_ok(), "rope_into failed: {:?}", result.err());

    exec.synchronize().expect("sync");
    let output = read_gpu_buffer(&output_gpu, size);

    assert_ne!(&output, &input, "RoPE should modify values");
}

// ============================================================================
// Strategy G.5: Ghost Loader - Weight Loading Tests
// ============================================================================

#[test]
fn test_g5_load_weights_synthetic() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let weights: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.001).collect();
    let result = exec.load_weights("synthetic.layer0.q", &weights);
    assert!(result.is_ok(), "load_weights failed: {:?}", result.err());
}

#[test]
fn test_g5_load_quantized_weights() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let q4k_data: Vec<u8> = vec![0x55; 144];
    let result = exec.load_quantized_weights("synthetic.q4k", &q4k_data);
    assert!(
        result.is_ok(),
        "load_quantized_weights failed: {:?}",
        result.err()
    );
}

#[test]
fn test_g5_clear_weights() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let weights: Vec<f32> = vec![1.0; 512];
    exec.load_weights("test.w1", &weights).expect("load");
    exec.load_weights("test.w2", &weights).expect("load");
    exec.clear_weights();
}

// ============================================================================
// Activation Functions - Direct GPU Tests
// ============================================================================

#[test]
fn test_g2_silu_gpu() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let n = 128u32;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.1).collect();

    let ctx = exec.context();
    let input_gpu = GpuBuffer::from_host(ctx, &input).expect("input");

    // silu_gpu returns a new buffer
    let result = exec.silu_gpu(&input_gpu, n);
    assert!(result.is_ok(), "SiLU failed: {:?}", result.err());

    let output_gpu = result.unwrap();
    let output = read_gpu_buffer(&output_gpu, n as usize);

    // SiLU(0) should be ~0
    let mid = (n / 2) as usize;
    assert!(
        output[mid].abs() < 0.1,
        "SiLU(0) should be ~0, got {}",
        output[mid]
    );
}

#[test]
fn test_g2_gelu_gpu() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let n = 128u32;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.1).collect();

    let ctx = exec.context();
    let buffer_gpu = GpuBuffer::from_host(ctx, &data).expect("buffer");

    // gelu_gpu is in-place
    let result = exec.gelu_gpu(&buffer_gpu, n);
    assert!(result.is_ok(), "GELU failed: {:?}", result.err());

    exec.synchronize().expect("sync");
    let output = read_gpu_buffer(&buffer_gpu, n as usize);

    // GELU(0) ≈ 0
    let mid = (n / 2) as usize;
    assert!(output[mid].abs() < 0.1, "GELU(0) should be ~0");
}

#[test]
fn test_g2_elementwise_mul() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let n = 256u32;
    let a: Vec<f32> = vec![2.0; n as usize];
    let b: Vec<f32> = vec![3.0; n as usize];

    let ctx = exec.context();
    let a_gpu = GpuBuffer::from_host(ctx, &a).expect("a");
    let b_gpu = GpuBuffer::from_host(ctx, &b).expect("b");

    // elementwise_mul_gpu returns a new buffer
    let result = exec.elementwise_mul_gpu(&a_gpu, &b_gpu, n);
    assert!(result.is_ok(), "elementwise_mul failed: {:?}", result.err());

    let c_gpu = result.unwrap();
    let output = read_gpu_buffer(&c_gpu, n as usize);

    for (i, &val) in output.iter().enumerate() {
        assert!(
            (val - 6.0).abs() < 0.01,
            "c[{}]: expected 6.0, got {}",
            i,
            val
        );
    }
}

#[test]
fn test_g2_residual_add() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let n = 256u32;
    let residual: Vec<f32> = vec![1.0; n as usize];
    let hidden: Vec<f32> = vec![2.0; n as usize];

    let ctx = exec.context();
    let res_gpu = GpuBuffer::from_host(ctx, &residual).expect("residual");
    let hid_gpu = GpuBuffer::from_host(ctx, &hidden).expect("hidden");

    // residual_add_gpu returns a new buffer
    let result = exec.residual_add_gpu(&res_gpu, &hid_gpu, n);
    assert!(result.is_ok(), "residual_add failed: {:?}", result.err());

    let out_gpu = result.unwrap();
    let output = read_gpu_buffer(&out_gpu, n as usize);

    for (i, &val) in output.iter().enumerate() {
        assert!(
            (val - 3.0).abs() < 0.01,
            "out[{}]: expected 3.0, got {}",
            i,
            val
        );
    }
}

// ============================================================================
// Memory Pool Tests
// ============================================================================

#[test]
fn test_memory_pool_stats() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    for i in 0..3 {
        let size = 1024 * (i + 1);
        let mut data: Vec<f32> = vec![1.0; size];
        let _ = exec.softmax(&mut data);
    }

    let stats = exec.pool_stats();
    assert!(
        stats.total_allocated > 0 || stats.pool_hits > 0 || stats.pool_misses > 0,
        "Memory pool should have activity"
    );
}

#[test]
fn test_staging_pool_stats() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    for i in 0..3 {
        let weights: Vec<f32> = vec![i as f32; 2048];
        let _ = exec.load_weights(&format!("staging_{}", i), &weights);
    }

    let stats = exec.staging_pool_stats();
    assert!(
        stats.total_allocated > 0 || stats.pool_hits > 0 || stats.pool_misses > 0,
        "Staging pool should have activity"
    );
}

// ============================================================================
// Profiler Tests
// ============================================================================

#[test]
fn test_profiler_enable_disable() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    assert!(!exec.is_profiling_enabled());
    exec.enable_profiling();
    assert!(exec.is_profiling_enabled());
    exec.disable_profiling();
    assert!(!exec.is_profiling_enabled());
}

// ============================================================================
// Device Info Tests
// ============================================================================

#[test]
fn test_device_info() {
    let exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let mem_info = exec.memory_info();
    assert!(mem_info.is_ok(), "memory_info failed: {:?}", mem_info.err());

    let (free, total) = mem_info.unwrap();
    assert!(total > 0, "Total memory should be positive");
    assert!(free <= total, "Free memory should not exceed total");
    assert!(
        total >= 20 * 1024 * 1024 * 1024,
        "Expected ~24GB, got {}",
        total
    );
}

#[test]
fn test_synchronize() {
    let exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let result = exec.synchronize();
    assert!(result.is_ok(), "synchronize failed: {:?}", result.err());
}

#[test]
fn test_is_available() {
    let available = CudaExecutor::is_available();
    let count = CudaExecutor::num_devices();

    if available {
        assert!(count > 0, "If available, should have at least 1 device");
    }
}

// ============================================================================
// SwiGLU Host Test - Strategy G.2 Direct Kernel Injection
// ============================================================================

#[test]
fn test_g2_swiglu_components_host() {
    // Test SiLU + elementwise mul manually to cover the host path
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let n = 64usize;

    // SwiGLU: output = SiLU(gate) * up
    let gate: Vec<f32> = vec![1.0; n];
    let up: Vec<f32> = vec![2.0; n];

    // First compute SiLU(gate)
    let mut silu_out = vec![0.0f32; n];
    let result = exec.silu_host(&gate, &mut silu_out);
    assert!(result.is_ok(), "silu_host failed: {:?}", result.err());

    // SiLU(1.0) ≈ 0.731
    assert!(
        silu_out[0] > 0.7 && silu_out[0] < 0.8,
        "SiLU(1.0) should be ~0.731"
    );

    // Then elementwise mul with up
    let mut output = vec![0.0f32; n];
    let result = exec.elementwise_mul_host(&silu_out, &up, &mut output);
    assert!(
        result.is_ok(),
        "elementwise_mul_host failed: {:?}",
        result.err()
    );

    // Result should be ~0.731 * 2.0 = ~1.462
    for (i, &val) in output.iter().enumerate() {
        assert!(
            val > 1.0 && val < 2.0,
            "output[{}]: expected ~1.46, got {}",
            i,
            val
        );
    }
}

// ============================================================================
// Q4K GEMV Tests - Cover quantized weight paths
// ============================================================================

#[test]
fn test_g2_q4k_gemv_kernel_type() {
    use realizar::cuda::KernelType;

    // Verify KernelType construction
    let kt = KernelType::Q4KGemv { k: 256, n: 64 };

    // This just verifies the enum variant exists and is constructed correctly
    match kt {
        KernelType::Q4KGemv { k, n } => {
            assert_eq!(k, 256);
            assert_eq!(n, 64);
        },
        _ => panic!("Wrong kernel type"),
    }
}

#[test]
fn test_g2_coalesced_q4k_gemv_kernel_type() {
    use realizar::cuda::KernelType;

    let kt = KernelType::CoalescedQ4KGemv { k: 512, n: 128 };

    match kt {
        KernelType::CoalescedQ4KGemv { k, n } => {
            assert_eq!(k, 512);
            assert_eq!(n, 128);
        },
        _ => panic!("Wrong kernel type"),
    }
}

// ============================================================================
// Attention Tests
// ============================================================================

#[test]
fn test_g2_flash_attention_basic() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let seq_len = 4u32;
    let head_dim = 32u32;
    let scale = 1.0 / (head_dim as f32).sqrt(); // 1/sqrt(d_k)

    let q: Vec<f32> = vec![0.1; (seq_len * head_dim) as usize];
    let k: Vec<f32> = vec![0.1; (seq_len * head_dim) as usize];
    let v: Vec<f32> = vec![1.0; (seq_len * head_dim) as usize];
    let mut output = vec![0.0f32; (seq_len * head_dim) as usize];

    let result = exec.flash_attention(&q, &k, &v, &mut output, seq_len, head_dim, scale, true);
    assert!(result.is_ok(), "flash_attention failed: {:?}", result.err());

    // Output should be non-zero
    let sum: f32 = output.iter().sum();
    assert!(sum.abs() > 0.01, "Attention output should be non-zero");
}

// ============================================================================
// Transformer Layer Tests - Strategy G.1 Synthetic Forward
// ============================================================================

#[test]
fn test_g1_transformer_layer_host() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let hidden_dim = 64usize;

    // Test host-side transformer layer with synthetic data
    let input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.01).collect();
    let gamma: Vec<f32> = vec![1.0; hidden_dim];

    // Just test RMSNorm + residual as host path
    let mut normalized = vec![0.0f32; hidden_dim];
    let result = exec.rmsnorm_host(&input, &gamma, &mut normalized, 1e-5);
    assert!(result.is_ok(), "rmsnorm_host failed: {:?}", result.err());

    // Normalized should have different magnitude
    let input_l2: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_l2: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (input_l2 - norm_l2).abs() > 0.01,
        "Host RMSNorm should change L2: input={}, output={}",
        input_l2,
        norm_l2
    );
}

#[test]
fn test_g1_silu_host() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let n = 64usize;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let mut output = vec![0.0f32; n];

    let result = exec.silu_host(&input, &mut output);
    assert!(result.is_ok(), "silu_host failed: {:?}", result.err());

    // SiLU(0) should be ~0
    assert!(output[n / 2].abs() < 0.1, "SiLU(0) should be ~0");
}

#[test]
fn test_g1_gelu_host() {
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let n = 64usize;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let mut output = vec![0.0f32; n];

    let result = exec.gelu_host(&input, &mut output);
    assert!(result.is_ok(), "gelu_host failed: {:?}", result.err());

    // GELU(0) ≈ 0
    assert!(output[n / 2].abs() < 0.1, "GELU(0) should be ~0");
}

// ============================================================================
// Strategy G.1: Synthetic Transformer Layer with Loaded Weights
// ============================================================================

/// Tiny synthetic model config for testing forward pass
struct SyntheticConfig {
    hidden_dim: usize,
    intermediate_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_layers: usize,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 64, // Small for fast tests
            intermediate_dim: 128,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 32,
            n_layers: 1,
        }
    }
}

/// Create synthetic Q4K quantized weights (144 bytes per 256 values)
fn create_q4k_weights(output_dim: usize, input_dim: usize) -> Vec<u8> {
    // Q4_K: 256 values per super-block, 144 bytes each
    let num_values = output_dim * input_dim;
    let num_superblocks = num_values.div_ceil(256);
    let bytes_per_block = 144;
    vec![0x55; num_superblocks * bytes_per_block]
}

#[test]
fn test_g1_synthetic_full_setup() {
    // Test setting up a synthetic model with all components
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let cfg = SyntheticConfig::default();

    // 1. Initialize workspace
    let result = exec.init_workspace(cfg.hidden_dim, cfg.intermediate_dim);
    assert!(result.is_ok(), "init_workspace failed: {:?}", result.err());

    // 2. Initialize KV cache
    let result = exec.init_kv_cache_gpu(
        cfg.n_layers,
        64, // max_seq_len
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
    );
    assert!(
        result.is_ok(),
        "init_kv_cache_gpu failed: {:?}",
        result.err()
    );

    // 3. Load synthetic weights for layer 0
    let layer_prefix = "model.layers.0";

    // Attention weights (Q, K, V, O projections)
    let q_size = cfg.n_heads * cfg.head_dim * cfg.hidden_dim;
    let kv_size = cfg.n_kv_heads * cfg.head_dim * cfg.hidden_dim;

    // Load as quantized Q4K weights
    let q_weight = create_q4k_weights(cfg.n_heads * cfg.head_dim, cfg.hidden_dim);
    let k_weight = create_q4k_weights(cfg.n_kv_heads * cfg.head_dim, cfg.hidden_dim);
    let v_weight = create_q4k_weights(cfg.n_kv_heads * cfg.head_dim, cfg.hidden_dim);
    let o_weight = create_q4k_weights(cfg.hidden_dim, cfg.n_heads * cfg.head_dim);

    exec.load_quantized_weights(&format!("{}.attn_q.weight", layer_prefix), &q_weight)
        .expect("load Q");
    exec.load_quantized_weights(&format!("{}.attn_k.weight", layer_prefix), &k_weight)
        .expect("load K");
    exec.load_quantized_weights(&format!("{}.attn_v.weight", layer_prefix), &v_weight)
        .expect("load V");
    exec.load_quantized_weights(&format!("{}.attn_output.weight", layer_prefix), &o_weight)
        .expect("load O");

    // FFN weights (gate, up, down)
    let gate_weight = create_q4k_weights(cfg.intermediate_dim, cfg.hidden_dim);
    let up_weight = create_q4k_weights(cfg.intermediate_dim, cfg.hidden_dim);
    let down_weight = create_q4k_weights(cfg.hidden_dim, cfg.intermediate_dim);

    exec.load_quantized_weights(&format!("{}.ffn_gate.weight", layer_prefix), &gate_weight)
        .expect("load gate");
    exec.load_quantized_weights(&format!("{}.ffn_up.weight", layer_prefix), &up_weight)
        .expect("load up");
    exec.load_quantized_weights(&format!("{}.ffn_down.weight", layer_prefix), &down_weight)
        .expect("load down");

    // 4. Load RMSNorm gamma weights (f32)
    let attn_norm: Vec<f32> = vec![1.0; cfg.hidden_dim];
    let ffn_norm: Vec<f32> = vec![1.0; cfg.hidden_dim];

    exec.load_weights(&format!("{}.attn_norm.weight", layer_prefix), &attn_norm)
        .expect("load attn_norm");
    exec.load_weights(&format!("{}.ffn_norm.weight", layer_prefix), &ffn_norm)
        .expect("load ffn_norm");

    // 5. Verify weights are cached
    assert!(exec.has_quantized_weights(&format!("{}.attn_q.weight", layer_prefix)));
    assert!(exec.has_quantized_weights(&format!("{}.ffn_gate.weight", layer_prefix)));
}

#[test]
fn test_g1_residual_add_into() {
    // Test residual_add_into (the pre-allocated version)
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let n = 256u32;
    let a: Vec<f32> = vec![1.0; n as usize];
    let b: Vec<f32> = vec![2.0; n as usize];

    let ctx = exec.context();
    let a_gpu = GpuBuffer::from_host(ctx, &a).expect("a");
    let b_gpu = GpuBuffer::from_host(ctx, &b).expect("b");
    let output_gpu = GpuBuffer::<f32>::new(ctx, n as usize).expect("output");

    let result = exec.residual_add_into(&a_gpu, &b_gpu, &output_gpu, n);
    assert!(
        result.is_ok(),
        "residual_add_into failed: {:?}",
        result.err()
    );

    exec.synchronize().expect("sync");
    let output = read_gpu_buffer(&output_gpu, n as usize);

    for (i, &val) in output.iter().enumerate() {
        assert!(
            (val - 3.0).abs() < 0.01,
            "out[{}]: expected 3.0, got {}",
            i,
            val
        );
    }
}

#[test]
fn test_g1_fused_swiglu_into() {
    // Test fused_swiglu_into (gate * SiLU(gate) * up in one kernel)
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let n = 128u32;
    let gate: Vec<f32> = vec![1.0; n as usize]; // SiLU(1) ≈ 0.731
    let up: Vec<f32> = vec![2.0; n as usize];

    let ctx = exec.context();
    let gate_gpu = GpuBuffer::from_host(ctx, &gate).expect("gate");
    let up_gpu = GpuBuffer::from_host(ctx, &up).expect("up");
    let output_gpu = GpuBuffer::<f32>::new(ctx, n as usize).expect("output");

    let result = exec.fused_swiglu_into(&gate_gpu, &up_gpu, &output_gpu, n);
    assert!(
        result.is_ok(),
        "fused_swiglu_into failed: {:?}",
        result.err()
    );

    exec.synchronize().expect("sync");
    let output = read_gpu_buffer(&output_gpu, n as usize);

    // SwiGLU(1, 2) = SiLU(1) * 2 ≈ 0.731 * 2 = 1.46
    for (i, &val) in output.iter().enumerate() {
        assert!(
            val > 1.0 && val < 2.0,
            "out[{}]: expected ~1.46, got {}",
            i,
            val
        );
    }
}

#[test]
fn test_g2_batched_rmsnorm() {
    // Test batched vectorized RMSNorm kernel
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    let batch_size = 4u32;
    let hidden_dim = 64u32;
    let total_elems = (batch_size * hidden_dim) as usize;

    let input: Vec<f32> = (0..total_elems).map(|i| (i as f32) * 0.01).collect();
    let gamma: Vec<f32> = vec![1.0; hidden_dim as usize];

    let ctx = exec.context();
    let input_gpu = GpuBuffer::from_host(ctx, &input).expect("input");
    let gamma_gpu = GpuBuffer::from_host(ctx, &gamma).expect("gamma");
    let output_gpu = GpuBuffer::<f32>::new(ctx, total_elems).expect("output");

    let result = exec.batched_rmsnorm_into(
        &input_gpu,
        &gamma_gpu,
        &output_gpu,
        batch_size,
        hidden_dim,
        1e-5,
    );
    assert!(
        result.is_ok(),
        "batched_rmsnorm_into failed: {:?}",
        result.err()
    );

    exec.synchronize().expect("sync");
    let output = read_gpu_buffer(&output_gpu, total_elems);

    // Each batch element should be normalized
    assert!(
        output.iter().all(|x| x.is_finite()),
        "All outputs should be finite"
    );
}

#[test]
fn test_g2_make_current() {
    // Test make_current for multi-threaded contexts
    let exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    // make_current should succeed (we're on the same thread)
    let result = exec.make_current();
    assert!(result.is_ok(), "make_current failed: {:?}", result.err());
}

#[test]
fn test_g2_kernel_caching() {
    // Test that kernels are cached after first use
    let mut exec = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {:?}", e);
            return;
        },
    };

    // Run softmax twice - second should use cached kernel
    let mut data1: Vec<f32> = vec![1.0; 64];
    let mut data2: Vec<f32> = vec![2.0; 64];

    let r1 = exec.softmax(&mut data1);
    let r2 = exec.softmax(&mut data2);

    assert!(
        r1.is_ok() && r2.is_ok(),
        "Cached kernel execution should succeed"
    );
}
