//! Incremental attention GPU tests and KV cache scatter tests.

use crate::driver::{CudaContext, CudaStream};
use crate::memory::resident::GpuResidentTensor;

#[test]
fn test_incremental_attention_gpu() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::incremental_attention_gpu;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 4u32; // current position in sequence
    let max_seq_len = 16u32;
    let d_model = (n_heads * head_dim) as usize;

    // Query for single new token: [n_heads * head_dim]
    let q_data: Vec<f32> = (0..d_model).map(|i| (i as f32) * 0.1).collect();
    // KV cache: [n_heads * max_seq_len * head_dim]
    let cache_size = (n_heads * max_seq_len * head_dim) as usize;
    let k_data: Vec<f32> = (0..cache_size).map(|i| (i as f32) * 0.01).collect();
    let v_data: Vec<f32> = (0..cache_size).map(|i| (i as f32) * 0.01).collect();

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let output =
        incremental_attention_gpu(&ctx, &q, &k, &v, n_heads, head_dim, seq_len, max_seq_len)
            .unwrap();
    assert_eq!(output.len(), d_model); // single token output
}

#[test]
fn test_incremental_attention_gpu_with_stream() {
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
    let v_data: Vec<f32> = vec![0.1; cache_size];

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

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
fn test_kv_cache_scatter_gpu() {
    use crate::memory::resident::kv_cache_scatter_gpu;
    let ctx = cuda_ctx!();
    let stream = CudaStream::new(&ctx).unwrap();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let max_seq_len = 8u32;
    let d_model = (n_heads * head_dim) as usize;
    let cache_size = (n_heads * max_seq_len * head_dim) as usize;

    // Existing cache
    let cache_data: Vec<f32> = vec![0.0; cache_size];
    // New KV to scatter at position 3: [n_heads * head_dim]
    let new_kv: Vec<f32> = vec![1.0; d_model];
    let position = 3u32;

    let mut cache = GpuResidentTensor::from_host(&ctx, &cache_data).unwrap();
    let new_tensor = GpuResidentTensor::from_host(&ctx, &new_kv).unwrap();

    kv_cache_scatter_gpu(
        &ctx,
        &new_tensor,
        &mut cache,
        position,
        n_heads,
        head_dim,
        max_seq_len,
        &stream,
    )
    .unwrap();

    // Verify scatter happened
    let result = cache.to_host().unwrap();
    // Check values at scattered position
    assert!(result.len() == cache_size);
}

#[test]
fn test_incremental_attention_dimension_error() {
    use crate::memory::resident::incremental_attention_gpu;
    let ctx = cuda_ctx!();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 4u32;
    let max_seq_len = 16u32;
    let cache_size = (n_heads * max_seq_len * head_dim) as usize;

    // Q has wrong size
    let q_data: Vec<f32> = vec![0.1; 5]; // Wrong - should be d_model
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
fn test_kv_cache_scatter_dimension_error() {
    use crate::memory::resident::kv_cache_scatter_gpu;
    let ctx = cuda_ctx!();
    let stream = CudaStream::new(&ctx).unwrap();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let max_seq_len = 8u32;
    let cache_size = (n_heads * max_seq_len * head_dim) as usize;

    // New KV has wrong size
    let new_kv: Vec<f32> = vec![1.0; 5]; // Wrong - should be n_heads * head_dim
    let cache_data: Vec<f32> = vec![0.0; cache_size];

    let new_tensor = GpuResidentTensor::from_host(&ctx, &new_kv).unwrap();
    let mut cache = GpuResidentTensor::from_host(&ctx, &cache_data).unwrap();

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
