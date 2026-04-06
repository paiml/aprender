use super::*;

#[test]
#[cfg(feature = "cuda")]
fn test_gemm_forward_basic() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // 2x2 @ 2x2 matrix multiplication
    // A = [[1, 2], [3, 4]]
    // B = [[5, 6], [7, 8]]
    // C = A @ B = [[19, 22], [43, 50]]
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
    let c_data: Vec<f32> = vec![0.0; 4];

    let a = GpuBuffer::from_host(&ctx, &a_data).expect("operation should succeed");
    let b = GpuBuffer::from_host(&ctx, &b_data).expect("operation should succeed");
    let mut c = GpuBuffer::from_host(&ctx, &c_data).expect("operation should succeed");

    gemm_forward(&a, &b, &mut c, 2, 2, 2, &stream).expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; 4];
    c.copy_to_host(&mut result).expect("operation should succeed");
    assert!((result[0] - 19.0).abs() < 1e-3, "C[0,0] should be 19, got {}", result[0]);
    assert!((result[1] - 22.0).abs() < 1e-3, "C[0,1] should be 22, got {}", result[1]);
    assert!((result[2] - 43.0).abs() < 1e-3, "C[1,0] should be 43, got {}", result[2]);
    assert!((result[3] - 50.0).abs() < 1e-3, "C[1,1] should be 50, got {}", result[3]);
}

#[test]
#[cfg(feature = "cuda")]
fn test_fused_swiglu_forward_basic() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // SwiGLU: output = SiLU(gate) * up
    let gate_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let up_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let n = gate_data.len() as u32;

    let gate = GpuBuffer::from_host(&ctx, &gate_data).expect("operation should succeed");
    let up = GpuBuffer::from_host(&ctx, &up_data).expect("operation should succeed");
    let output_data = vec![0.0f32; n as usize];
    let mut output = GpuBuffer::from_host(&ctx, &output_data).expect("operation should succeed");

    fused_swiglu_forward(&gate, &up, &mut output, n, &stream).expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; n as usize];
    output.copy_to_host(&mut result).expect("operation should succeed");

    // With up=1, output should equal SiLU(gate)
    // SiLU(x) = x * sigmoid(x) > 0 for x > 0
    for (i, &r) in result.iter().enumerate() {
        assert!(r > 0.0, "SwiGLU output should be positive for positive gate, got {r} at {i}");
    }
    // Output should increase with gate (monotonic for positive inputs)
    assert!(result[3] > result[2]);
    assert!(result[2] > result[1]);
    assert!(result[1] > result[0]);
}

#[test]
#[cfg(feature = "cuda")]
fn test_gemm_forward_2x2() {
    // Test GEMM with a simple 2x2 case for verification
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // A = [[1, 2], [3, 4]] (2x2, row-major)
    // B = [[1, 0], [0, 1]] (2x2, identity)
    // C = A @ B = A
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let c_data: Vec<f32> = vec![0.0; 4];

    let a = GpuBuffer::from_host(&ctx, &a_data).expect("operation should succeed");
    let b = GpuBuffer::from_host(&ctx, &b_data).expect("operation should succeed");
    let mut c = GpuBuffer::from_host(&ctx, &c_data).expect("operation should succeed");

    gemm_forward(&a, &b, &mut c, 2, 2, 2, &stream).expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; 4];
    c.copy_to_host(&mut result).expect("operation should succeed");

    // C should equal A (since B is identity)
    for (i, (&r, &expected)) in result.iter().zip(a_data.iter()).enumerate() {
        assert!((r - expected).abs() < 1e-4, "GEMM mismatch at {i}: got {r}, expected {expected}");
    }
}

/// Test GEMM with 0.5B model dimensions (hidden=896, intermediate=4864)
/// Reproduces CUDA_ERROR_ILLEGAL_ADDRESS (code 700) seen in production.
#[test]
#[cfg(feature = "cuda")]
fn test_gemm_forward_896x896() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // Q projection: seq_len=1 x hidden=896 @ hidden=896 x hidden=896
    let m = 1u32;
    let k = 896u32;
    let n = 896u32;

    let a_data = vec![0.01f32; (m * k) as usize];
    let b_data = vec![0.01f32; (k * n) as usize];

    let a = GpuBuffer::from_host(&ctx, &a_data).expect("operation should succeed");
    let b = GpuBuffer::from_host(&ctx, &b_data).expect("operation should succeed");
    let mut c = GpuBuffer::new(&ctx, (m * n) as usize).expect("operation should succeed");

    eprintln!("GEMM {m}x{k} @ {k}x{n}...");
    gemm_forward(&a, &b, &mut c, m, k, n, &stream).expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; (m * n) as usize];
    c.copy_to_host(&mut result).expect("operation should succeed");

    // Expected: each element = sum(0.01 * 0.01 for 896 terms) = 896 * 0.0001 = 0.0896
    let expected = 896.0 * 0.01 * 0.01;
    assert!(
        (result[0] - expected).abs() < 0.01,
        "GEMM 896x896 result[0] = {}, expected ~{expected}",
        result[0]
    );
    eprintln!("GEMM {m}x{k} @ {k}x{n} OK, result[0]={}", result[0]);
}

#[test]
#[cfg(feature = "cuda")]
fn test_gemm_forward_896x4864() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // Gate/Up projection: seq_len=1 x hidden=896 @ hidden=896 x intermediate=4864
    let m = 1u32;
    let k = 896u32;
    let n = 4864u32;

    let a_data = vec![0.01f32; (m * k) as usize];
    let b_data = vec![0.01f32; (k * n) as usize];

    let a = GpuBuffer::from_host(&ctx, &a_data).expect("operation should succeed");
    let b = GpuBuffer::from_host(&ctx, &b_data).expect("operation should succeed");
    let mut c = GpuBuffer::new(&ctx, (m * n) as usize).expect("operation should succeed");

    eprintln!("GEMM {m}x{k} @ {k}x{n}...");
    gemm_forward(&a, &b, &mut c, m, k, n, &stream).expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; (m * n) as usize];
    c.copy_to_host(&mut result).expect("operation should succeed");

    let expected = 896.0 * 0.01 * 0.01;
    assert!(
        (result[0] - expected).abs() < 0.01,
        "GEMM 896x4864 result[0] = {}, expected ~{expected}",
        result[0]
    );
    eprintln!("GEMM {m}x{k} @ {k}x{n} OK, result[0]={}", result[0]);
}

#[test]
#[cfg(feature = "cuda")]
fn test_gemm_forward_4864x896() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // Down projection: seq_len=1 x intermediate=4864 @ intermediate=4864 x hidden=896
    let m = 1u32;
    let k = 4864u32;
    let n = 896u32;

    let a_data = vec![0.01f32; (m * k) as usize];
    let b_data = vec![0.01f32; (k * n) as usize];

    let a = GpuBuffer::from_host(&ctx, &a_data).expect("operation should succeed");
    let b = GpuBuffer::from_host(&ctx, &b_data).expect("operation should succeed");
    let mut c = GpuBuffer::new(&ctx, (m * n) as usize).expect("operation should succeed");

    eprintln!("GEMM {m}x{k} @ {k}x{n}...");
    gemm_forward(&a, &b, &mut c, m, k, n, &stream).expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; (m * n) as usize];
    c.copy_to_host(&mut result).expect("operation should succeed");

    let expected = 4864.0 * 0.01 * 0.01;
    assert!(
        (result[0] - expected).abs() < 0.01,
        "GEMM 4864x896 result[0] = {}, expected ~{expected}",
        result[0]
    );
    eprintln!("GEMM {m}x{k} @ {k}x{n} OK, result[0]={}", result[0]);
}

/// Full forward pass simulation with 0.5B dimensions
/// This tests the exact sequence of operations that fails in production.
#[test]
#[cfg(feature = "cuda")]
fn test_full_forward_0_5b_dims() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let seq_len = 1u32;
    let hidden = 896u32;
    let intermediate = 4864u32;

    eprintln!("=== Full forward simulation with 0.5B dims ===");

    // Allocate buffers
    let input_data = vec![0.01f32; (seq_len * hidden) as usize];
    let norm_w = vec![1.0f32; hidden as usize];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let norm_weight = GpuBuffer::from_host(&ctx, &norm_w).expect("operation should succeed");
    let mut norm_out =
        GpuBuffer::new(&ctx, (seq_len * hidden) as usize).expect("operation should succeed");

    // Step 1: RMSNorm
    eprintln!("Step 1: RMSNorm seq={seq_len} hidden={hidden}");
    rms_norm_forward(&input, &norm_weight, &mut norm_out, seq_len, hidden, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");
    eprintln!("  OK");

    // Step 2: Q-GEMM
    let w_q_data = vec![0.001f32; (hidden * hidden) as usize];
    let w_q = GpuBuffer::from_host(&ctx, &w_q_data).expect("operation should succeed");
    let mut q_out =
        GpuBuffer::new(&ctx, (seq_len * hidden) as usize).expect("operation should succeed");

    eprintln!("Step 2: Q-GEMM {seq_len}x{hidden} @ {hidden}x{hidden}");
    gemm_forward(&norm_out, &w_q, &mut q_out, seq_len, hidden, hidden, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");
    eprintln!("  OK");

    // Step 3: Gate-GEMM
    let w_gate_data = vec![0.001f32; (hidden * intermediate) as usize];
    let w_gate = GpuBuffer::from_host(&ctx, &w_gate_data).expect("operation should succeed");
    let mut gate_out =
        GpuBuffer::new(&ctx, (seq_len * intermediate) as usize).expect("operation should succeed");

    eprintln!("Step 3: Gate-GEMM {seq_len}x{hidden} @ {hidden}x{intermediate}");
    gemm_forward(&norm_out, &w_gate, &mut gate_out, seq_len, hidden, intermediate, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");
    eprintln!("  OK");

    // Step 4: Up-GEMM
    let w_up_data = vec![0.001f32; (hidden * intermediate) as usize];
    let w_up = GpuBuffer::from_host(&ctx, &w_up_data).expect("operation should succeed");
    let mut up_out =
        GpuBuffer::new(&ctx, (seq_len * intermediate) as usize).expect("operation should succeed");

    eprintln!("Step 4: Up-GEMM {seq_len}x{hidden} @ {hidden}x{intermediate}");
    gemm_forward(&norm_out, &w_up, &mut up_out, seq_len, hidden, intermediate, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");
    eprintln!("  OK");

    // Step 5: Fused SwiGLU
    let n_swiglu = seq_len * intermediate;
    let mut swiglu_out = GpuBuffer::new(&ctx, n_swiglu as usize).expect("operation should succeed");

    eprintln!("Step 5: FusedSwiGLU n={n_swiglu}");
    fused_swiglu_forward(&gate_out, &up_out, &mut swiglu_out, n_swiglu, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");
    eprintln!("  OK");

    // Step 6: Down-GEMM
    let w_down_data = vec![0.001f32; (intermediate * hidden) as usize];
    let w_down = GpuBuffer::from_host(&ctx, &w_down_data).expect("operation should succeed");
    let mut ffn_out =
        GpuBuffer::new(&ctx, (seq_len * hidden) as usize).expect("operation should succeed");

    eprintln!("Step 6: Down-GEMM {seq_len}x{intermediate} @ {intermediate}x{hidden}");
    gemm_forward(&swiglu_out, &w_down, &mut ffn_out, seq_len, intermediate, hidden, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");
    eprintln!("  OK");

    eprintln!("=== Full forward simulation PASSED ===");
}

// === BF16-precision GPU GEMM tests ===

#[test]
#[cfg(feature = "cuda")]
fn test_gemm_forward_bf16_basic() {
    use crate::autograd::precision::{bf16_truncate, gemm_bf16_reference};
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};
    let _ = bf16_truncate; // suppress unused import if not needed directly

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // 2x2 @ 2x2 with exact bf16 values
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
    let c_data: Vec<f32> = vec![0.0; 4];

    let a = GpuBuffer::from_host(&ctx, &a_data).expect("operation should succeed");
    let b = GpuBuffer::from_host(&ctx, &b_data).expect("operation should succeed");
    let mut c = GpuBuffer::from_host(&ctx, &c_data).expect("operation should succeed");

    gemm_forward_bf16(&a, &b, &mut c, 2, 2, 2, &stream).expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; 4];
    c.copy_to_host(&mut result).expect("operation should succeed");

    // With exact bf16 values (small integers), result matches fp32
    let expected = gemm_bf16_reference(&a_data, &b_data, 2, 2, 2);
    for i in 0..4 {
        assert!(
            (result[i] - expected[i]).abs() < 1e-3,
            "BF16 GEMM GPU vs CPU mismatch at [{i}]: GPU={}, CPU={}",
            result[i],
            expected[i]
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_gemm_forward_bf16_precision_matches_reference() {
    use crate::autograd::precision::gemm_bf16_reference;
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // Non-exact values that will exhibit bf16 truncation
    let m = 4u32;
    let k = 8u32;
    let n = 4u32;
    let a_data: Vec<f32> = (0..m * k).map(|i| 0.01 * (i as f32 + 1.0)).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| 0.01 * (i as f32 + 1.0)).collect();

    let a = GpuBuffer::from_host(&ctx, &a_data).expect("operation should succeed");
    let b = GpuBuffer::from_host(&ctx, &b_data).expect("operation should succeed");
    let mut c = GpuBuffer::from_host(&ctx, &vec![0.0f32; (m * n) as usize])
        .expect("operation should succeed");

    gemm_forward_bf16(&a, &b, &mut c, m, k, n, &stream).expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; (m * n) as usize];
    c.copy_to_host(&mut result).expect("operation should succeed");

    let expected = gemm_bf16_reference(&a_data, &b_data, m as usize, k as usize, n as usize);
    for i in 0..(m * n) as usize {
        assert!(
            (result[i] - expected[i]).abs() < 1e-3,
            "BF16 GEMM GPU/CPU mismatch at [{i}]: GPU={}, CPU={}",
            result[i],
            expected[i]
        );
    }
}

/// Test fused SwiGLU with production-sized n (seq_len * intermediate_size).
/// Reproduces PTX JIT failure seen at n=510720.
#[test]
#[cfg(feature = "cuda")]
fn test_fused_swiglu_large_n() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");

    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // Production dimensions: seq_len=105 * intermediate=4864 = 510720
    let n = 510_720u32;
    eprintln!("FusedSwiGLU large n={n}...");

    let gate_data = vec![0.5f32; n as usize];
    let up_data = vec![1.0f32; n as usize];

    let gate = GpuBuffer::from_host(&ctx, &gate_data).expect("operation should succeed");
    let up = GpuBuffer::from_host(&ctx, &up_data).expect("operation should succeed");
    let mut output = GpuBuffer::new(&ctx, n as usize).expect("operation should succeed");

    fused_swiglu_forward(&gate, &up, &mut output, n, &stream).expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; n as usize];
    output.copy_to_host(&mut result).expect("operation should succeed");

    // SiLU(0.5) * 1.0 = 0.5 * sigmoid(0.5) ≈ 0.5 * 0.6225 ≈ 0.3112
    let expected = 0.5 * (1.0 / (1.0 + (-0.5f64).exp()));
    assert!(
        (f64::from(result[0]) - expected).abs() < 0.01,
        "SwiGLU large n result[0] = {}, expected ~{expected}",
        result[0]
    );
    eprintln!("FusedSwiGLU n={n} OK, result[0]={}", result[0]);
}
