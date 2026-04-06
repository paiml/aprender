use super::*;

// =============================================================================
// Contract C-RESADD-001: residual_add_forward
// =============================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_contract_residual_add_gpu_equals_cpu() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let a_data: Vec<f32> = vec![1.0, 2.5, -3.0, 4.0, 0.0, -1.5, 7.0, 8.0];
    let b_data: Vec<f32> = vec![0.5, -2.5, 3.0, -4.0, 1.0, 1.5, -7.0, 2.0];
    let n = a_data.len();

    // CPU reference
    let expected: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a + b).collect();

    let a = GpuBuffer::from_host(&ctx, &a_data).expect("operation should succeed");
    let b = GpuBuffer::from_host(&ctx, &b_data).expect("operation should succeed");
    let mut output = GpuBuffer::new(&ctx, n).expect("operation should succeed");

    residual_add_forward(&a, &b, &mut output, n as u32, &stream).expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; n];
    output.copy_to_host(&mut result).expect("operation should succeed");

    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "residual_add mismatch at {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_contract_residual_add_large() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // Test with larger buffer crossing multiple thread blocks (>256 elements)
    let n = 1024;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let b_data: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.005).collect();
    let expected: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a + b).collect();

    let a = GpuBuffer::from_host(&ctx, &a_data).expect("operation should succeed");
    let b = GpuBuffer::from_host(&ctx, &b_data).expect("operation should succeed");
    let mut output = GpuBuffer::new(&ctx, n).expect("operation should succeed");

    residual_add_forward(&a, &b, &mut output, n as u32, &stream).expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; n];
    output.copy_to_host(&mut result).expect("operation should succeed");

    let max_err: f32 =
        result.iter().zip(expected.iter()).map(|(g, e)| (g - e).abs()).fold(0.0f32, f32::max);
    assert!(max_err < 1e-5, "residual_add large max error: {max_err}");
}

// =============================================================================
// Contract C-ELMUL-001: elementwise_mul_forward
// =============================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_contract_elementwise_mul_gpu_equals_cpu() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5];
    let b_data: Vec<f32> = vec![2.0, 0.5, -1.0, 3.0, 4.0, 2.0];
    let n = a_data.len();

    let expected: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a * b).collect();

    let a = GpuBuffer::from_host(&ctx, &a_data).expect("operation should succeed");
    let b = GpuBuffer::from_host(&ctx, &b_data).expect("operation should succeed");
    let mut output = GpuBuffer::new(&ctx, n).expect("operation should succeed");

    elementwise_mul_forward(&a, &b, &mut output, n as u32, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; n];
    output.copy_to_host(&mut result).expect("operation should succeed");

    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "elementwise_mul mismatch at {i}: got {got}, expected {exp}"
        );
    }
}

// =============================================================================
// Contract C-SCALE-001: scale_forward
// =============================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_contract_scale_gpu_equals_cpu() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 5.0, 10.0];
    let scale_val = 0.125f32; // 1/sqrt(64), typical attention scale
    let n = input_data.len();

    let expected: Vec<f32> = input_data.iter().map(|x| x * scale_val).collect();

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let mut output = GpuBuffer::new(&ctx, n).expect("operation should succeed");

    scale_forward(&input, &mut output, scale_val, n as u32, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; n];
    output.copy_to_host(&mut result).expect("operation should succeed");

    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "scale mismatch at {i}: got {got}, expected {exp}");
    }
}

// =============================================================================
// Contract C-I2B-001: interleaved_to_batched_forward
// =============================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_contract_interleaved_to_batched() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // 2 seq positions, 2 heads, 3 head_dim
    // Interleaved: [seq, n_heads * head_dim] = [2, 6]
    // Row 0: [h0d0, h0d1, h0d2, h1d0, h1d1, h1d2]
    // Row 1: [h0d0, h0d1, h0d2, h1d0, h1d1, h1d2]
    let seq_len = 2u32;
    let n_heads = 2u32;
    let head_dim = 3u32;
    let total = (seq_len * n_heads * head_dim) as usize;

    // input[s, h*head_dim + d]
    let input_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // seq=0: head0=[1,2,3], head1=[4,5,6]
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // seq=1: head0=[7,8,9], head1=[10,11,12]
    ];

    // Expected batched: [n_heads, seq, head_dim]
    // head0: [[1,2,3], [7,8,9]]
    // head1: [[4,5,6], [10,11,12]]
    let expected: Vec<f32> = vec![
        1.0, 2.0, 3.0, 7.0, 8.0, 9.0, // head 0
        4.0, 5.0, 6.0, 10.0, 11.0, 12.0, // head 1
    ];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let mut output = GpuBuffer::new(&ctx, total).expect("operation should succeed");

    interleaved_to_batched_forward(&input, &mut output, seq_len, n_heads, head_dim, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; total];
    output.copy_to_host(&mut result).expect("operation should succeed");

    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "interleaved_to_batched mismatch at {i}: got {got}, expected {exp}"
        );
    }
}

// =============================================================================
// Contract C-BTRANS-001: batched_transpose_forward
// =============================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_contract_batched_transpose() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // 2 batches, 2 rows, 3 cols
    // batch 0: [[1,2,3],[4,5,6]] → transposed: [[1,4],[2,5],[3,6]]
    // batch 1: [[7,8,9],[10,11,12]] → transposed: [[7,10],[8,11],[9,12]]
    let batch = 2u32;
    let rows = 2u32;
    let cols = 3u32;
    let total = (batch * rows * cols) as usize;

    let input_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch 1
    ];

    // After transpose [batch, cols, rows]:
    let expected: Vec<f32> = vec![
        1.0, 4.0, 2.0, 5.0, 3.0, 6.0, // batch 0 transposed
        7.0, 10.0, 8.0, 11.0, 9.0, 12.0, // batch 1 transposed
    ];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let mut output = GpuBuffer::new(&ctx, total).expect("operation should succeed");

    batched_transpose_forward(&input, &mut output, batch, rows, cols, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; total];
    output.copy_to_host(&mut result).expect("operation should succeed");

    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "batched_transpose mismatch at {i}: got {got}, expected {exp}"
        );
    }
}

// =============================================================================
// Contract C-B2I-001: batched_to_interleaved_forward
// =============================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_contract_batched_to_interleaved_roundtrip() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // Roundtrip: interleaved → batched → interleaved should be identity
    let seq_len = 4u32;
    let n_heads = 3u32;
    let head_dim = 2u32;
    let total = (seq_len * n_heads * head_dim) as usize;

    let original: Vec<f32> = (0..total).map(|i| i as f32 + 1.0).collect();

    let input = GpuBuffer::from_host(&ctx, &original).expect("operation should succeed");
    let mut batched = GpuBuffer::new(&ctx, total).expect("operation should succeed");
    let mut roundtrip = GpuBuffer::new(&ctx, total).expect("operation should succeed");

    // interleaved → batched
    interleaved_to_batched_forward(&input, &mut batched, seq_len, n_heads, head_dim, &stream)
        .expect("operation should succeed");

    // batched → interleaved
    batched_to_interleaved_forward(&batched, &mut roundtrip, seq_len, n_heads, head_dim, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; total];
    roundtrip.copy_to_host(&mut result).expect("operation should succeed");

    for (i, (&got, &exp)) in result.iter().zip(original.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "roundtrip mismatch at {i}: got {got}, expected {exp}");
    }
}

// =============================================================================
// Contract C-GQAEXP-001: expand_kv_heads
// =============================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_contract_expand_kv_heads() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // 2 KV heads, heads_per_kv=2 → 4 total heads
    // Each head has 2 seq × 3 head_dim = 6 elements
    let num_kv_heads = 2;
    let heads_per_kv = 2;
    let num_heads = num_kv_heads * heads_per_kv;
    let elems_per_head = 6; // seq_len * head_dim

    // KV head 0: [1,2,3,4,5,6], KV head 1: [7,8,9,10,11,12]
    let src_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // kv head 0
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // kv head 1
    ];

    // Expected: head 0 = kv0, head 1 = kv0, head 2 = kv1, head 3 = kv1
    let expected: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // head 0 (from kv 0)
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // head 1 (from kv 0)
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // head 2 (from kv 1)
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // head 3 (from kv 1)
    ];

    let src = GpuBuffer::from_host(&ctx, &src_data).expect("operation should succeed");
    let mut dst =
        GpuBuffer::new(&ctx, num_heads * elems_per_head).expect("operation should succeed");

    expand_kv_heads(&src, &mut dst, num_kv_heads, heads_per_kv, elems_per_head, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; num_heads * elems_per_head];
    dst.copy_to_host(&mut result).expect("operation should succeed");

    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "expand_kv_heads mismatch at {i}: got {got}, expected {exp}"
        );
    }
}

// =============================================================================
// Contract C-B4DGEMM-001: batched_4d_gemm_forward
// =============================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_contract_batched_4d_gemm_single_head() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // Single batch, single head: 2x3 @ 3x2 → 2x2
    let batch = 1u32;
    let heads = 1u32;
    let m = 2u32;
    let k = 3u32;
    let n = 2u32;

    // A = [[1,2,3],[4,5,6]]
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // B = [[1,0],[0,1],[1,1]]
    let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    // C = A @ B = [[4,5],[10,11]]
    let expected: Vec<f32> = vec![4.0, 5.0, 10.0, 11.0];

    let a = GpuBuffer::from_host(&ctx, &a_data).expect("operation should succeed");
    let b = GpuBuffer::from_host(&ctx, &b_data).expect("operation should succeed");
    let mut c = GpuBuffer::new(&ctx, (m * n) as usize).expect("operation should succeed");

    batched_4d_gemm_forward(&a, &b, &mut c, batch, heads, m, n, k, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; (m * n) as usize];
    c.copy_to_host(&mut result).expect("operation should succeed");

    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "batched_4d_gemm mismatch at {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_contract_batched_4d_gemm_multi_head() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // 1 batch, 2 heads, 2x2 @ 2x2
    let batch = 1u32;
    let heads = 2u32;
    let m = 2u32;
    let k = 2u32;
    let n = 2u32;

    // Head 0: A=[[1,0],[0,1]] (identity), B=[[5,6],[7,8]] → C=[[5,6],[7,8]]
    // Head 1: A=[[2,0],[0,2]] (2*I), B=[[1,2],[3,4]] → C=[[2,4],[6,8]]
    let a_data: Vec<f32> = vec![
        1.0, 0.0, 0.0, 1.0, // head 0: identity
        2.0, 0.0, 0.0, 2.0, // head 1: 2*identity
    ];
    let b_data: Vec<f32> = vec![
        5.0, 6.0, 7.0, 8.0, // head 0
        1.0, 2.0, 3.0, 4.0, // head 1
    ];
    let expected: Vec<f32> = vec![
        5.0, 6.0, 7.0, 8.0, // head 0: I @ B = B
        2.0, 4.0, 6.0, 8.0, // head 1: 2I @ B = 2B
    ];

    let total = (batch * heads * m * n) as usize;
    let a = GpuBuffer::from_host(&ctx, &a_data).expect("operation should succeed");
    let b = GpuBuffer::from_host(&ctx, &b_data).expect("operation should succeed");
    let mut c = GpuBuffer::new(&ctx, total).expect("operation should succeed");

    batched_4d_gemm_forward(&a, &b, &mut c, batch, heads, m, n, k, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; total];
    c.copy_to_host(&mut result).expect("operation should succeed");

    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "batched_4d_gemm multi-head mismatch at {i}: got {got}, expected {exp}"
        );
    }
}

// =============================================================================
// Contract C-BSMAX-001: batched_softmax_forward
// =============================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_contract_batched_softmax_rows_sum_to_one() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // 3 rows × 4 cols
    let total_rows = 3u32;
    let row_size = 4u32;
    let total = (total_rows * row_size) as usize;

    let input_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, // row 0
        -1.0, 0.0, 1.0, 2.0, // row 1
        0.0, 0.0, 0.0, 0.0, // row 2 (uniform)
    ];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let mut output = GpuBuffer::new(&ctx, total).expect("operation should succeed");

    batched_softmax_forward(&input, &mut output, total_rows, row_size, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; total];
    output.copy_to_host(&mut result).expect("operation should succeed");

    // Check each row sums to 1.0
    for r in 0..total_rows as usize {
        let row_start = r * row_size as usize;
        let row_sum: f32 = result[row_start..row_start + row_size as usize].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-4, "softmax row {r} sums to {row_sum}, expected 1.0");

        // All values should be positive
        for j in 0..row_size as usize {
            assert!(
                result[row_start + j] > 0.0,
                "softmax row {r} col {j} is non-positive: {}",
                result[row_start + j]
            );
        }
    }

    // Row 2 (uniform input) should have equal probabilities
    let uniform_expected = 0.25f32;
    for j in 0..row_size as usize {
        assert!(
            (result[8 + j] - uniform_expected).abs() < 1e-4,
            "uniform row col {j}: got {}, expected {uniform_expected}",
            result[8 + j]
        );
    }

    // Row 0: softmax([1,2,3,4]) — values should be monotonically increasing
    assert!(result[0] < result[1]);
    assert!(result[1] < result[2]);
    assert!(result[2] < result[3]);
}

#[test]
#[cfg(feature = "cuda")]
fn test_contract_batched_softmax_numerical_stability() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // Large values that would overflow exp() without max-subtraction trick
    let total_rows = 1u32;
    let row_size = 4u32;
    let total = (total_rows * row_size) as usize;

    let input_data: Vec<f32> = vec![1000.0, 1001.0, 1002.0, 1003.0];

    let input = GpuBuffer::from_host(&ctx, &input_data).expect("operation should succeed");
    let mut output = GpuBuffer::new(&ctx, total).expect("operation should succeed");

    batched_softmax_forward(&input, &mut output, total_rows, row_size, &stream)
        .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; total];
    output.copy_to_host(&mut result).expect("operation should succeed");

    // Should not produce NaN or Inf despite large inputs
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "softmax produced non-finite at {i}: {v}");
        assert!(v > 0.0, "softmax produced non-positive at {i}: {v}");
    }

    let row_sum: f32 = result.iter().sum();
    assert!(
        (row_sum - 1.0).abs() < 1e-4,
        "softmax with large inputs sums to {row_sum}, expected 1.0"
    );
}

// =============================================================================
// Contract C-ATTN-001: End-to-end attention pipeline
// =============================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_contract_attention_pipeline_layout_roundtrip() {
    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    if !cuda_available() {
        return;
    }

    let ctx = CudaContext::new(0).expect("operation should succeed");
    let ctx = std::sync::Arc::new(ctx);
    init_forward_kernel_cache(ctx.clone()).expect("operation should succeed");
    let stream = CudaStream::new(&ctx).expect("operation should succeed");

    // Simulate the attention layout pipeline:
    // interleaved → batched → transpose → transpose → batched → interleaved
    // Double transpose should be identity
    let seq_len = 4u32;
    let n_heads = 2u32;
    let head_dim = 8u32;
    let total = (seq_len * n_heads * head_dim) as usize;

    let original: Vec<f32> = (0..total).map(|i| (i as f32 + 1.0) * 0.1).collect();

    let input = GpuBuffer::from_host(&ctx, &original).expect("operation should succeed");
    let mut batched = GpuBuffer::new(&ctx, total).expect("operation should succeed");
    let mut transposed = GpuBuffer::new(&ctx, total).expect("operation should succeed");
    let mut transposed_back = GpuBuffer::new(&ctx, total).expect("operation should succeed");
    let _batched_back: GpuBuffer<f32> =
        GpuBuffer::new(&ctx, total).expect("operation should succeed");
    let mut roundtrip = GpuBuffer::new(&ctx, total).expect("operation should succeed");

    // interleaved → batched [n_heads, seq, head_dim]
    interleaved_to_batched_forward(&input, &mut batched, seq_len, n_heads, head_dim, &stream)
        .expect("operation should succeed");

    // transpose [n_heads, seq, head_dim] → [n_heads, head_dim, seq]
    batched_transpose_forward(&batched, &mut transposed, n_heads, seq_len, head_dim, &stream)
        .expect("operation should succeed");

    // transpose back [n_heads, head_dim, seq] → [n_heads, seq, head_dim]
    batched_transpose_forward(
        &transposed,
        &mut transposed_back,
        n_heads,
        head_dim,
        seq_len,
        &stream,
    )
    .expect("operation should succeed");

    // Verify batched == transposed_back (double transpose is identity)
    stream.synchronize().expect("operation should succeed");
    let mut batched_data = vec![0.0f32; total];
    let mut transposed_back_data = vec![0.0f32; total];
    batched.copy_to_host(&mut batched_data).expect("operation should succeed");
    transposed_back.copy_to_host(&mut transposed_back_data).expect("operation should succeed");

    for (i, (&got, &exp)) in transposed_back_data.iter().zip(batched_data.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "double transpose mismatch at {i}: got {got}, expected {exp}"
        );
    }

    // batched → interleaved (full roundtrip)
    batched_to_interleaved_forward(
        &transposed_back,
        &mut roundtrip,
        seq_len,
        n_heads,
        head_dim,
        &stream,
    )
    .expect("operation should succeed");
    stream.synchronize().expect("operation should succeed");

    let mut result = vec![0.0f32; total];
    roundtrip.copy_to_host(&mut result).expect("operation should succeed");

    for (i, (&got, &exp)) in result.iter().zip(original.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "full layout roundtrip mismatch at {i}: got {got}, expected {exp}"
        );
    }
}
