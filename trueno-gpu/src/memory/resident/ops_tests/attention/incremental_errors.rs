//! Error propagation tests for incremental attention, async variants,
//! stream variants, and KV cache scatter edge cases.

use crate::driver::{CudaContext, CudaStream};
use crate::memory::resident::GpuResidentTensor;

// ============================================================================
// PMAT-018: Additional attention.rs coverage tests
// ============================================================================

#[test]
fn test_incremental_attention_v_cache_error() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::incremental_attention_gpu;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 4u32;
    let max_seq_len = 16u32;
    let d_model = (n_heads * head_dim) as usize;
    let cache_size = (n_heads * max_seq_len * head_dim) as usize;

    let q_data: Vec<f32> = vec![0.1; d_model];
    let k_data: Vec<f32> = vec![0.1; cache_size];
    // V cache has wrong size
    let v_data: Vec<f32> = vec![0.1; 10]; // Wrong!

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let result =
        incremental_attention_gpu(&ctx, &q, &k, &v, n_heads, head_dim, seq_len, max_seq_len);
    assert!(result.is_err());
}

#[test]
fn test_incremental_attention_k_cache_error() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::incremental_attention_gpu;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 4u32;
    let max_seq_len = 16u32;
    let d_model = (n_heads * head_dim) as usize;
    let cache_size = (n_heads * max_seq_len * head_dim) as usize;

    let q_data: Vec<f32> = vec![0.1; d_model];
    // K cache has wrong size
    let k_data: Vec<f32> = vec![0.1; 10]; // Wrong!
    let v_data: Vec<f32> = vec![0.1; cache_size];

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let result =
        incremental_attention_gpu(&ctx, &q, &k, &v, n_heads, head_dim, seq_len, max_seq_len);
    assert!(result.is_err());
}

#[test]
fn test_incremental_attention_seq_exceeds_max() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::incremental_attention_gpu;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 20u32; // exceeds max_seq_len!
    let max_seq_len = 16u32;
    let d_model = (n_heads * head_dim) as usize;
    let cache_size = (n_heads * max_seq_len * head_dim) as usize;

    let q_data: Vec<f32> = vec![0.1; d_model];
    let k_data: Vec<f32> = vec![0.1; cache_size];
    let v_data: Vec<f32> = vec![0.1; cache_size];

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let result =
        incremental_attention_gpu(&ctx, &q, &k, &v, n_heads, head_dim, seq_len, max_seq_len);
    assert!(result.is_err());
}

#[test]
fn test_incremental_attention_empty_sequence() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::incremental_attention_gpu;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 0u32; // Empty sequence
    let max_seq_len = 16u32;
    let d_model = (n_heads * head_dim) as usize;
    let cache_size = (n_heads * max_seq_len * head_dim) as usize;

    let q_data: Vec<f32> = vec![0.1; d_model];
    let k_data: Vec<f32> = vec![0.1; cache_size];
    let v_data: Vec<f32> = vec![0.1; cache_size];

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    // Empty sequence should return zeros
    let output =
        incremental_attention_gpu(&ctx, &q, &k, &v, n_heads, head_dim, seq_len, max_seq_len)
            .unwrap();
    assert_eq!(output.len(), d_model);
}

#[test]
fn test_incremental_attention_gpu_async() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::incremental_attention_gpu_async;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 4u32;
    let max_seq_len = 16u32;
    let d_model = (n_heads * head_dim) as usize;
    let cache_size = (n_heads * max_seq_len * head_dim) as usize;

    let q_data: Vec<f32> = (0..d_model).map(|i| (i as f32) * 0.1).collect();
    let k_data: Vec<f32> = (0..cache_size).map(|i| (i as f32) * 0.01).collect();
    let v_data: Vec<f32> = (0..cache_size).map(|i| (i as f32) * 0.01).collect();

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let (output, stream) =
        incremental_attention_gpu_async(&ctx, &q, &k, &v, n_heads, head_dim, seq_len, max_seq_len)
            .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(output.len(), d_model);
}

#[test]
fn test_incremental_attention_gpu_async_empty_seq() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::incremental_attention_gpu_async;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 0u32; // Empty sequence
    let max_seq_len = 16u32;
    let d_model = (n_heads * head_dim) as usize;
    let cache_size = (n_heads * max_seq_len * head_dim) as usize;

    let q_data: Vec<f32> = vec![0.1; d_model];
    let k_data: Vec<f32> = vec![0.1; cache_size];
    let v_data: Vec<f32> = vec![0.1; cache_size];

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    // Empty sequence should return zeros + stream
    let (output, _stream) =
        incremental_attention_gpu_async(&ctx, &q, &k, &v, n_heads, head_dim, seq_len, max_seq_len)
            .unwrap();
    assert_eq!(output.len(), d_model);
}

#[test]
fn test_incremental_attention_gpu_async_q_error() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::incremental_attention_gpu_async;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 4u32;
    let max_seq_len = 16u32;
    let cache_size = (n_heads * max_seq_len * head_dim) as usize;

    // Q has wrong size
    let q_data: Vec<f32> = vec![0.1; 5]; // Wrong!
    let k_data: Vec<f32> = vec![0.1; cache_size];
    let v_data: Vec<f32> = vec![0.1; cache_size];

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let result =
        incremental_attention_gpu_async(&ctx, &q, &k, &v, n_heads, head_dim, seq_len, max_seq_len);
    assert!(result.is_err());
}

#[test]
fn test_incremental_attention_gpu_async_kv_cache_error() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::incremental_attention_gpu_async;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 4u32;
    let max_seq_len = 16u32;
    let d_model = (n_heads * head_dim) as usize;

    let q_data: Vec<f32> = vec![0.1; d_model];
    // K/V cache has wrong size
    let k_data: Vec<f32> = vec![0.1; 10]; // Wrong!
    let v_data: Vec<f32> = vec![0.1; 10]; // Wrong!

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let result =
        incremental_attention_gpu_async(&ctx, &q, &k, &v, n_heads, head_dim, seq_len, max_seq_len);
    assert!(result.is_err());
}

#[test]
fn test_incremental_attention_gpu_async_seq_exceeds_max() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::incremental_attention_gpu_async;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 20u32; // Exceeds max!
    let max_seq_len = 16u32;
    let d_model = (n_heads * head_dim) as usize;
    let cache_size = (n_heads * max_seq_len * head_dim) as usize;

    let q_data: Vec<f32> = vec![0.1; d_model];
    let k_data: Vec<f32> = vec![0.1; cache_size];
    let v_data: Vec<f32> = vec![0.1; cache_size];

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let result =
        incremental_attention_gpu_async(&ctx, &q, &k, &v, n_heads, head_dim, seq_len, max_seq_len);
    assert!(result.is_err());
}

#[test]
fn test_incremental_attention_with_stream_q_error() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::incremental_attention_gpu_with_stream;
    clear_kernel_cache();
    let ctx = cuda_ctx!();
    let stream = CudaStream::new(&ctx).unwrap();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 4u32;
    let max_seq_len = 16u32;
    let cache_size = (n_heads * max_seq_len * head_dim) as usize;

    // Q has wrong size
    let q_data: Vec<f32> = vec![0.1; 5]; // Wrong!
    let k_data: Vec<f32> = vec![0.1; cache_size];
    let v_data: Vec<f32> = vec![0.1; cache_size];

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let result = incremental_attention_gpu_with_stream(
        &ctx,
        &q,
        &k,
        &v,
        n_heads,
        head_dim,
        seq_len,
        max_seq_len,
        &stream,
    );
    assert!(result.is_err());
}

#[test]
fn test_incremental_attention_with_stream_k_cache_error() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::incremental_attention_gpu_with_stream;
    clear_kernel_cache();
    let ctx = cuda_ctx!();
    let stream = CudaStream::new(&ctx).unwrap();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 4u32;
    let max_seq_len = 16u32;
    let d_model = (n_heads * head_dim) as usize;
    let cache_size = (n_heads * max_seq_len * head_dim) as usize;

    let q_data: Vec<f32> = vec![0.1; d_model];
    // K cache has wrong size
    let k_data: Vec<f32> = vec![0.1; 10]; // Wrong!
    let v_data: Vec<f32> = vec![0.1; cache_size];

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let result = incremental_attention_gpu_with_stream(
        &ctx,
        &q,
        &k,
        &v,
        n_heads,
        head_dim,
        seq_len,
        max_seq_len,
        &stream,
    );
    assert!(result.is_err());
}

#[test]
fn test_incremental_attention_with_stream_v_cache_error() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::incremental_attention_gpu_with_stream;
    clear_kernel_cache();
    let ctx = cuda_ctx!();
    let stream = CudaStream::new(&ctx).unwrap();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 4u32;
    let max_seq_len = 16u32;
    let d_model = (n_heads * head_dim) as usize;
    let cache_size = (n_heads * max_seq_len * head_dim) as usize;

    let q_data: Vec<f32> = vec![0.1; d_model];
    let k_data: Vec<f32> = vec![0.1; cache_size];
    // V cache has wrong size
    let v_data: Vec<f32> = vec![0.1; 10]; // Wrong!

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let result = incremental_attention_gpu_with_stream(
        &ctx,
        &q,
        &k,
        &v,
        n_heads,
        head_dim,
        seq_len,
        max_seq_len,
        &stream,
    );
    assert!(result.is_err());
}

#[test]
fn test_incremental_attention_with_stream_seq_exceeds_max() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::incremental_attention_gpu_with_stream;
    clear_kernel_cache();
    let ctx = cuda_ctx!();
    let stream = CudaStream::new(&ctx).unwrap();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 20u32; // Exceeds max!
    let max_seq_len = 16u32;
    let d_model = (n_heads * head_dim) as usize;
    let cache_size = (n_heads * max_seq_len * head_dim) as usize;

    let q_data: Vec<f32> = vec![0.1; d_model];
    let k_data: Vec<f32> = vec![0.1; cache_size];
    let v_data: Vec<f32> = vec![0.1; cache_size];

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let result = incremental_attention_gpu_with_stream(
        &ctx,
        &q,
        &k,
        &v,
        n_heads,
        head_dim,
        seq_len,
        max_seq_len,
        &stream,
    );
    assert!(result.is_err());
}

#[test]
fn test_incremental_attention_with_stream_empty_seq() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::incremental_attention_gpu_with_stream;
    clear_kernel_cache();
    let ctx = cuda_ctx!();
    let stream = CudaStream::new(&ctx).unwrap();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 0u32; // Empty sequence
    let max_seq_len = 16u32;
    let d_model = (n_heads * head_dim) as usize;
    let cache_size = (n_heads * max_seq_len * head_dim) as usize;

    let q_data: Vec<f32> = vec![0.1; d_model];
    let k_data: Vec<f32> = vec![0.1; cache_size];
    let v_data: Vec<f32> = vec![0.1; cache_size];

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    // Empty sequence should return zeros
    let output = incremental_attention_gpu_with_stream(
        &ctx,
        &q,
        &k,
        &v,
        n_heads,
        head_dim,
        seq_len,
        max_seq_len,
        &stream,
    )
    .unwrap();
    assert_eq!(output.len(), d_model);
}

#[test]
fn test_kv_cache_scatter_cache_size_error() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::kv_cache_scatter_gpu;
    clear_kernel_cache();
    let ctx = cuda_ctx!();
    let stream = CudaStream::new(&ctx).unwrap();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let max_seq_len = 8u32;
    let d_model = (n_heads * head_dim) as usize;

    // Cache has wrong size
    let cache_data: Vec<f32> = vec![0.0; 100]; // Wrong! Should be n_heads * max_seq_len * head_dim
    let new_kv: Vec<f32> = vec![1.0; d_model];

    let mut cache = GpuResidentTensor::from_host(&ctx, &cache_data).unwrap();
    let new_tensor = GpuResidentTensor::from_host(&ctx, &new_kv).unwrap();

    let result = kv_cache_scatter_gpu(
        &ctx,
        &new_tensor,
        &mut cache,
        3,
        n_heads,
        head_dim,
        max_seq_len,
        &stream,
    );
    assert!(result.is_err());
}

#[test]
fn test_kv_cache_scatter_position_exceeds_max() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::kv_cache_scatter_gpu;
    clear_kernel_cache();
    let ctx = cuda_ctx!();
    let stream = CudaStream::new(&ctx).unwrap();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let max_seq_len = 8u32;
    let d_model = (n_heads * head_dim) as usize;
    let cache_size = (n_heads * max_seq_len * head_dim) as usize;

    let cache_data: Vec<f32> = vec![0.0; cache_size];
    let new_kv: Vec<f32> = vec![1.0; d_model];
    let position = 10u32; // Exceeds max_seq_len!

    let mut cache = GpuResidentTensor::from_host(&ctx, &cache_data).unwrap();
    let new_tensor = GpuResidentTensor::from_host(&ctx, &new_kv).unwrap();

    let result = kv_cache_scatter_gpu(
        &ctx,
        &new_tensor,
        &mut cache,
        position,
        n_heads,
        head_dim,
        max_seq_len,
        &stream,
    );
    assert!(result.is_err());
}
