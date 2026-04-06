//! Incremental attention for autoregressive decoding and KV cache operations.
//!
//! Contains GPU-resident implementations for single-token attention against
//! the full KV cache, with synchronous, external-stream, and async variants.

#[cfg(feature = "cuda")]
use super::super::cache::compile_lock_launch;
#[cfg(feature = "cuda")]
use super::super::GpuResidentTensor;
#[cfg(feature = "cuda")]
use crate::driver::{CudaContext, CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use crate::error::Result;

// ============================================================================
// Validation and launch helpers (reduce cognitive complexity)
// ============================================================================

/// Validated parameters for incremental attention kernel launch.
#[cfg(feature = "cuda")]
struct IncrementalAttentionParams {
    q_expected: usize,
}

/// Validate Q, K/V cache, and seq_len parameters for incremental attention.
///
/// Returns `q_expected` (= n_heads * head_dim) on success.
#[cfg(feature = "cuda")]
fn validate_incremental_attention(
    q: &GpuResidentTensor<f32>,
    k_cache: &GpuResidentTensor<f32>,
    v_cache: &GpuResidentTensor<f32>,
    n_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
) -> Result<IncrementalAttentionParams> {
    let q_expected = (n_heads * head_dim) as usize;
    if q.len() != q_expected {
        return Err(crate::GpuError::InvalidParameter(format!(
            "Q has {} elements, expected {} (n_heads={}, head_dim={})",
            q.len(),
            q_expected,
            n_heads,
            head_dim
        )));
    }

    let cache_expected = (n_heads * max_seq_len * head_dim) as usize;
    if k_cache.len() != cache_expected {
        return Err(crate::GpuError::InvalidParameter(format!(
            "K cache has {} elements, expected {} (n_heads={}, max_seq_len={}, head_dim={})",
            k_cache.len(),
            cache_expected,
            n_heads,
            max_seq_len,
            head_dim
        )));
    }
    if v_cache.len() != cache_expected {
        return Err(crate::GpuError::InvalidParameter(format!(
            "V cache has {} elements, expected {}",
            v_cache.len(),
            cache_expected
        )));
    }

    if seq_len > max_seq_len {
        return Err(crate::GpuError::InvalidParameter(format!(
            "seq_len ({}) exceeds max_seq_len ({})",
            seq_len, max_seq_len
        )));
    }

    Ok(IncrementalAttentionParams { q_expected })
}

/// Launch the incremental attention kernel on the given stream.
///
/// Handles module lock acquisition, argument marshalling, and kernel dispatch.
#[cfg(feature = "cuda")]
fn launch_incremental_attention_kernel(
    ctx: &CudaContext,
    q: &GpuResidentTensor<f32>,
    k_cache: &GpuResidentTensor<f32>,
    v_cache: &GpuResidentTensor<f32>,
    output: &GpuBuffer<f32>,
    n_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
    stream: &CudaStream,
) -> Result<()> {
    use crate::kernels::{IncrementalAttentionKernel, Kernel};

    let kernel = IncrementalAttentionKernel::new(max_seq_len, head_dim, n_heads);
    let ptx = kernel.emit_ptx();
    let cache_key = format!(
        "incremental_attention:{}:{}:{}",
        max_seq_len, head_dim, n_heads
    );
    let config = LaunchConfig {
        grid: (n_heads, 1, 1),
        block: (32, 1, 1), // One warp
        shared_mem: 0,
    };

    let q_ptr = q.as_ptr();
    let k_ptr = k_cache.as_ptr();
    let v_ptr = v_cache.as_ptr();
    let out_ptr = output.as_ptr();
    let seq_len_val = seq_len;

    let mut args: [*mut std::ffi::c_void; 5] = [
        std::ptr::addr_of!(q_ptr) as *mut _,
        std::ptr::addr_of!(k_ptr) as *mut _,
        std::ptr::addr_of!(v_ptr) as *mut _,
        std::ptr::addr_of!(out_ptr) as *mut _,
        std::ptr::addr_of!(seq_len_val) as *mut _,
    ];

    compile_lock_launch(
        ctx,
        stream,
        &cache_key,
        &ptx,
        kernel.name(),
        &config,
        &mut args,
    )?;

    Ok(())
}

// ============================================================================
// Incremental Attention (Autoregressive Decoder)
// ============================================================================

/// WAPR-PERF-013: Incremental attention for autoregressive decoding
///
/// Computes attention for a single query token against the full KV cache.
/// Designed for GPU-resident KV caches with zero D2H transfers.
///
/// # Memory Layout (Head-First)
///
/// - `q`: `[n_heads, head_dim]` - query for current position (1 token)
/// - `k_cache`: `[n_heads, max_seq_len, head_dim]` - cached keys (head-first)
/// - `v_cache`: `[n_heads, max_seq_len, head_dim]` - cached values (head-first)
/// - output: `[n_heads, head_dim]` - attention output
///
/// # Arguments
///
/// * `ctx` - CUDA context
/// * `q` - Query tensor `[n_heads * head_dim]`
/// * `k_cache` - Key cache `[n_heads * max_seq_len * head_dim]`
/// * `v_cache` - Value cache `[n_heads * max_seq_len * head_dim]`
/// * `n_heads` - Number of attention heads
/// * `head_dim` - Dimension per head
/// * `seq_len` - Current sequence length (tokens in cache)
/// * `max_seq_len` - Maximum sequence length (cache capacity)
///
/// # Returns
///
/// Output tensor `[n_heads * head_dim]` (same shape as Q)
#[cfg(feature = "cuda")]
pub fn incremental_attention_gpu(
    ctx: &CudaContext,
    q: &GpuResidentTensor<f32>,
    k_cache: &GpuResidentTensor<f32>,
    v_cache: &GpuResidentTensor<f32>,
    n_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
) -> Result<GpuResidentTensor<f32>> {
    let params = validate_incremental_attention(
        q,
        k_cache,
        v_cache,
        n_heads,
        head_dim,
        seq_len,
        max_seq_len,
    )?;

    // Handle empty sequence (no attention needed)
    if seq_len == 0 {
        let zeros = vec![0.0f32; params.q_expected];
        return GpuResidentTensor::from_host(ctx, &zeros);
    }

    // Allocate output: [n_heads, head_dim]
    let output = GpuBuffer::new(ctx, params.q_expected)?;
    let stream = CudaStream::new(ctx)?;

    launch_incremental_attention_kernel(
        ctx,
        q,
        k_cache,
        v_cache,
        &output,
        n_heads,
        head_dim,
        seq_len,
        max_seq_len,
        &stream,
    )?;

    // WAPR-PERF-014: MUST sync before returning since stream goes out of scope
    // Without sync, kernel may not complete before output is used (UB!)
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}

/// WAPR-PERF-014: Incremental attention with external stream (no stream creation)
///
/// Same as `incremental_attention_gpu` but uses caller-provided stream instead of
/// creating a new one. This eliminates stream creation overhead (~5-10us per call).
///
/// # Use Case
///
/// When running attention in a loop (autoregressive decoding), use a single shared
/// stream for all operations to avoid creating ~40 streams per token.
///
/// # Arguments
///
/// * `stream` - Caller-provided CUDA stream (reuse across operations)
/// * Other args same as `incremental_attention_gpu`
#[cfg(feature = "cuda")]
pub fn incremental_attention_gpu_with_stream(
    ctx: &CudaContext,
    q: &GpuResidentTensor<f32>,
    k_cache: &GpuResidentTensor<f32>,
    v_cache: &GpuResidentTensor<f32>,
    n_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
    stream: &CudaStream,
) -> Result<GpuResidentTensor<f32>> {
    let params = validate_incremental_attention(
        q,
        k_cache,
        v_cache,
        n_heads,
        head_dim,
        seq_len,
        max_seq_len,
    )?;

    // Handle empty sequence
    if seq_len == 0 {
        let zeros = vec![0.0f32; params.q_expected];
        return GpuResidentTensor::from_host(ctx, &zeros);
    }

    // Allocate output
    let output = GpuBuffer::new(ctx, params.q_expected)?;

    launch_incremental_attention_kernel(
        ctx,
        q,
        k_cache,
        v_cache,
        &output,
        n_heads,
        head_dim,
        seq_len,
        max_seq_len,
        stream,
    )?;

    // NO SYNC - uses caller's stream for pipelining
    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}

/// WAPR-PERF-013: Async incremental attention with explicit stream return
///
/// Same as `incremental_attention_gpu` but returns the stream for caller-controlled
/// synchronization. Use this in autoregressive loops to avoid ghost syncs.
///
/// # Point 149 Compliance
///
/// This function launches the kernel without synchronizing. The caller MUST:
/// 1. Chain dependent operations on the same stream, OR
/// 2. Call `stream.synchronize()` before reading the output
///
/// # Returns
///
/// Tuple of (output tensor, stream) - stream must be synchronized before reading output
#[cfg(feature = "cuda")]
pub fn incremental_attention_gpu_async(
    ctx: &CudaContext,
    q: &GpuResidentTensor<f32>,
    k_cache: &GpuResidentTensor<f32>,
    v_cache: &GpuResidentTensor<f32>,
    n_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
) -> Result<(GpuResidentTensor<f32>, CudaStream)> {
    let params = validate_incremental_attention(
        q,
        k_cache,
        v_cache,
        n_heads,
        head_dim,
        seq_len,
        max_seq_len,
    )?;

    // Handle empty sequence
    if seq_len == 0 {
        let zeros = vec![0.0f32; params.q_expected];
        let output = GpuResidentTensor::from_host(ctx, &zeros)?;
        let stream = CudaStream::new(ctx)?;
        return Ok((output, stream));
    }

    // Allocate output
    let output = GpuBuffer::new(ctx, params.q_expected)?;
    let stream = CudaStream::new(ctx)?;

    launch_incremental_attention_kernel(
        ctx,
        q,
        k_cache,
        v_cache,
        &output,
        n_heads,
        head_dim,
        seq_len,
        max_seq_len,
        &stream,
    )?;

    // NO SYNC - caller controls synchronization (Point 149)
    Ok((GpuResidentTensor::from_buffer_internal(output, 1), stream))
}

/// WAPR-PERF-013: Scatter interleaved K/V to head-first cache slot
///
/// Writes a single position's K or V projection directly into the head-first
/// cache layout without intermediate conversion.
///
/// # Memory Layout
///
/// - Source: `[n_heads * head_dim]` (interleaved, from GEMV output)
/// - Cache: `[n_heads, max_seq_len, head_dim]` (head-first)
/// - Position `pos` is written to `cache[head, pos, :]` for all heads
///
/// # Performance
///
/// - Single kernel launch (no conversion overhead)
/// - Coalesced writes (threads write contiguous elements per head)
/// - Can be chained on same stream as GEMV (no sync needed)
///
/// # Arguments
///
/// * `ctx` - CUDA context
/// * `src` - Source tensor `[n_heads * head_dim]` (interleaved)
/// * `cache` - Target cache buffer `[n_heads * max_seq_len * head_dim]`
/// * `pos` - Sequence position to write
/// * `n_heads` - Number of attention heads
/// * `head_dim` - Dimension per head
/// * `max_seq_len` - Maximum sequence length (cache capacity)
/// * `stream` - CUDA stream for async execution
#[cfg(feature = "cuda")]
pub fn kv_cache_scatter_gpu(
    ctx: &CudaContext,
    src: &GpuResidentTensor<f32>,
    cache: &mut GpuResidentTensor<f32>,
    pos: u32,
    n_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    stream: &CudaStream,
) -> Result<()> {
    use crate::kernels::{Kernel, KvCacheScatterKernel};

    // Validate source size
    let src_expected = (n_heads * head_dim) as usize;
    if src.len() != src_expected {
        return Err(crate::GpuError::InvalidParameter(format!(
            "Source has {} elements, expected {} (n_heads={}, head_dim={})",
            src.len(),
            src_expected,
            n_heads,
            head_dim
        )));
    }

    // Validate cache size
    let cache_expected = (n_heads * max_seq_len * head_dim) as usize;
    if cache.len() != cache_expected {
        return Err(crate::GpuError::InvalidParameter(format!(
            "Cache has {} elements, expected {} (n_heads={}, max_seq_len={}, head_dim={})",
            cache.len(),
            cache_expected,
            n_heads,
            max_seq_len,
            head_dim
        )));
    }

    // Validate position
    if pos >= max_seq_len {
        return Err(crate::GpuError::InvalidParameter(format!(
            "Position {} >= max_seq_len {}",
            pos, max_seq_len
        )));
    }

    // Build and cache kernel
    let kernel = KvCacheScatterKernel::new(n_heads, head_dim, max_seq_len);
    let ptx = kernel.emit_ptx();
    let cache_key = format!("kv_scatter:{}:{}:{}", n_heads, head_dim, max_seq_len);
    // Launch config: one block per head, head_dim threads per block
    let config = LaunchConfig {
        grid: (n_heads, 1, 1),
        block: (head_dim.min(256), 1, 1), // Cap at 256 threads
        shared_mem: 0,
    };

    let src_ptr = src.as_ptr();
    let cache_ptr = cache.as_ptr();

    let mut args: [*mut std::ffi::c_void; 5] = [
        std::ptr::addr_of!(src_ptr) as *mut _,
        std::ptr::addr_of!(cache_ptr) as *mut _,
        std::ptr::addr_of!(pos) as *mut _,
        std::ptr::addr_of!(head_dim) as *mut _,
        std::ptr::addr_of!(max_seq_len) as *mut _,
    ];

    compile_lock_launch(
        ctx,
        stream,
        &cache_key,
        &ptx,
        kernel.name(),
        &config,
        &mut args,
    )?;

    // NO SYNC - caller chains operations (Point 149)
    Ok(())
}
