//! Batched multi-head attention implementations.
//!
//! Contains standard per-head and optimized all-heads-parallel attention.

#[cfg(feature = "cuda")]
use super::super::GpuResidentTensor;
#[cfg(feature = "cuda")]
use crate::driver::GpuBuffer;
#[cfg(feature = "cuda")]
use crate::error::Result;

// Note: Batched4DGemmKernel available for optimized multi-head attention (future)
#[allow(unused_imports)]
#[cfg(feature = "cuda")]
use crate::kernels::Batched4DGemmKernel;

#[cfg(feature = "cuda")]
use super::helpers::{
    batched_gemm, batched_scale_all, batched_softmax_all, batched_to_interleaved_all,
    batched_transpose_all, copy_head_to_output, extract_single_head, interleaved_to_batched_all,
    transpose_matrix,
};

#[cfg(feature = "cuda")]
use crate::driver::CudaContext;

// ============================================================================
// Batched Multi-Head Attention (GPU-Resident)
// ============================================================================

/// Batched multi-head attention that stays on GPU
///
/// Computes: output = softmax(Q @ K^T / sqrt(d_k)) @ V
/// All operations happen on GPU with ZERO intermediate host transfers.
///
/// This simplified version uses standard matmul operations per head.
/// The key benefit is ZERO host↔device transfers during computation.
///
/// # Arguments
///
/// * `ctx` - CUDA context
/// * `q` - Query tensor [seq_len * d_model] flattened
/// * `k` - Key tensor [seq_len * d_model] flattened
/// * `v` - Value tensor [seq_len * d_model] flattened
/// * `n_heads` - Number of attention heads
/// * `head_dim` - Dimension per head
/// * `seq_len` - Sequence length
///
/// # Returns
///
/// Output tensor [seq_len * d_model] still on GPU
///
/// # Citations
///
/// - [Vaswani2017] Attention Is All You Need - original multi-head attention
/// - [Dao2022] FlashAttention - fused attention for memory efficiency
#[cfg(feature = "cuda")]
pub fn batched_multihead_attention(
    ctx: &CudaContext,
    q: &GpuResidentTensor<f32>,
    k: &GpuResidentTensor<f32>,
    v: &GpuResidentTensor<f32>,
    n_heads: u32,
    head_dim: u32,
    seq_len: u32,
) -> Result<GpuResidentTensor<f32>> {
    let d_model = (n_heads * head_dim) as usize;
    let expected_size = (seq_len as usize) * d_model;

    // Validate input dimensions
    if q.len() != expected_size {
        return Err(crate::GpuError::InvalidParameter(format!(
            "Q has {} elements, expected {} (seq_len={}, d_model={})",
            q.len(),
            expected_size,
            seq_len,
            d_model
        )));
    }
    if k.len() != expected_size || v.len() != expected_size {
        return Err(crate::GpuError::InvalidParameter(
            "K and V must have same size as Q".to_string(),
        ));
    }

    // Proper multi-head attention: process each head independently
    // This involves more kernel launches but produces correct results.
    // Batched optimization can be added later.
    //
    // For each head h:
    //   1. Extract Q_h, K_h, V_h from interleaved layout
    //   2. Transpose K_h: [seq_len, head_dim] -> [head_dim, seq_len]
    //   3. Scores = Q_h @ K_h^T: [seq_len, head_dim] @ [head_dim, seq_len] = [seq_len, seq_len]
    //   4. Scale and softmax
    //   5. Output_h = Attn @ V_h: [seq_len, seq_len] @ [seq_len, head_dim] = [seq_len, head_dim]
    //   6. Copy Output_h to output at head h position

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Allocate output buffer
    let output_buffer = GpuBuffer::new(ctx, expected_size)?;

    let debug_attn = std::env::var("WHISPER_DEBUG_ATTN").is_ok();

    for h in 0..n_heads {
        let out_h = compute_single_head_attention(
            ctx, q, k, v, h, seq_len, n_heads, head_dim, scale, debug_attn,
        )?;

        // Copy out_h to output at head h position
        copy_head_to_output(ctx, &output_buffer, &out_h, h, seq_len, n_heads, head_dim)?;
    }

    Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
}

/// Compute attention for a single head.
///
/// Extracts Q_h, K_h, V_h, then computes softmax(Q_h @ K_h^T / sqrt(d_k)) @ V_h.
#[cfg(feature = "cuda")]
fn compute_single_head_attention(
    ctx: &CudaContext,
    q: &GpuResidentTensor<f32>,
    k: &GpuResidentTensor<f32>,
    v: &GpuResidentTensor<f32>,
    h: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    scale: f32,
    debug_attn: bool,
) -> Result<GpuResidentTensor<f32>> {
    // Extract head h from Q, K, V
    let q_h = extract_single_head(ctx, q, h, seq_len, n_heads, head_dim)?;
    let k_h = extract_single_head(ctx, k, h, seq_len, n_heads, head_dim)?;
    let v_h = extract_single_head(ctx, v, h, seq_len, n_heads, head_dim)?;

    maybe_debug_head_inputs(debug_attn, h, &q_h, &k_h, &v_h)?;

    // Transpose K_h: [seq_len, head_dim] -> [head_dim, seq_len]
    let kt_h = transpose_matrix(ctx, &k_h.buffer, seq_len, head_dim)?;
    let kt_tensor = GpuResidentTensor::from_buffer_internal(kt_h, 1);

    // Q_h @ K_h^T: [seq_len, head_dim] @ [head_dim, seq_len] = [seq_len, seq_len]
    let scores_h = q_h.matmul(ctx, &kt_tensor, seq_len, seq_len, head_dim)?;

    maybe_debug_scores(debug_attn, h, &scores_h)?;

    // Scale and softmax
    let scaled_h = scores_h.scale(ctx, scale)?;
    let attn_h = scaled_h.softmax(ctx, seq_len)?;

    maybe_debug_attention_weights(debug_attn, h, &attn_h, seq_len)?;

    // Attn @ V_h: [seq_len, seq_len] @ [seq_len, head_dim] = [seq_len, head_dim]
    let out_h = attn_h.matmul(ctx, &v_h, seq_len, head_dim, seq_len)?;

    maybe_debug_head_output(debug_attn, h, &out_h)?;

    Ok(out_h)
}

/// Conditionally debug-print mean of Q, K, V head tensors (only for head 0).
#[cfg(feature = "cuda")]
fn maybe_debug_head_inputs(
    debug_attn: bool,
    h: u32,
    q_h: &GpuResidentTensor<f32>,
    k_h: &GpuResidentTensor<f32>,
    v_h: &GpuResidentTensor<f32>,
) -> Result<()> {
    if !(debug_attn && h == 0) {
        return Ok(());
    }
    let q_host = q_h.peek_host()?;
    let k_host = k_h.peek_host()?;
    let v_host = v_h.peek_host()?;
    eprintln!(
        "[DEBUG-ATTN] head 0: Q_h mean={:.6}, K_h mean={:.6}, V_h mean={:.6}",
        q_host.iter().sum::<f32>() / q_host.len() as f32,
        k_host.iter().sum::<f32>() / k_host.len() as f32,
        v_host.iter().sum::<f32>() / v_host.len() as f32
    );
    Ok(())
}

/// Conditionally debug-print score statistics (only for head 0).
#[cfg(feature = "cuda")]
fn maybe_debug_scores(debug_attn: bool, h: u32, scores_h: &GpuResidentTensor<f32>) -> Result<()> {
    if !(debug_attn && h == 0) {
        return Ok(());
    }
    let scores_host = scores_h.peek_host()?;
    eprintln!(
        "[DEBUG-ATTN] head 0: scores mean={:.6}, max={:.6}",
        scores_host.iter().sum::<f32>() / scores_host.len() as f32,
        scores_host.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    Ok(())
}

/// Conditionally debug-print attention weight statistics (only for head 0).
#[cfg(feature = "cuda")]
fn maybe_debug_attention_weights(
    debug_attn: bool,
    h: u32,
    attn_h: &GpuResidentTensor<f32>,
    seq_len: u32,
) -> Result<()> {
    if !(debug_attn && h == 0) {
        return Ok(());
    }
    let attn_host = attn_h.peek_host()?;
    let first_row_sum: f32 = attn_host[..seq_len as usize].iter().sum();
    eprintln!(
        "[DEBUG-ATTN] head 0: attn first_row_sum={:.6}, mean={:.6}",
        first_row_sum,
        attn_host.iter().sum::<f32>() / attn_host.len() as f32
    );
    Ok(())
}

/// Conditionally debug-print output head statistics (only for head 0).
#[cfg(feature = "cuda")]
fn maybe_debug_head_output(debug_attn: bool, h: u32, out_h: &GpuResidentTensor<f32>) -> Result<()> {
    if !(debug_attn && h == 0) {
        return Ok(());
    }
    let out_host = out_h.peek_host()?;
    eprintln!(
        "[DEBUG-ATTN] head 0: out mean={:.6}, std={:.6}",
        out_host.iter().sum::<f32>() / out_host.len() as f32,
        (out_host.iter().map(|v| v.powi(2)).sum::<f32>() / out_host.len() as f32).sqrt()
    );
    Ok(())
}

/// Batched multi-head attention optimized for all heads in parallel (WAPR-PERF-008)
///
/// Reduces kernel launches from 54 (6 heads × 9 ops) to 9 by batching all heads.
/// Uses grid.z = n_heads for parallel head processing.
///
/// # Memory Layout
/// - Input Q, K, V: [seq_len, d_model] interleaved (d_model = n_heads * head_dim)
/// - Internal: [n_heads, seq_len, head_dim] batched for parallel processing
/// - Output: [seq_len, d_model] interleaved
#[cfg(feature = "cuda")]
pub fn batched_multihead_attention_optimized(
    ctx: &CudaContext,
    q: &GpuResidentTensor<f32>,
    k: &GpuResidentTensor<f32>,
    v: &GpuResidentTensor<f32>,
    n_heads: u32,
    head_dim: u32,
    seq_len: u32,
) -> Result<GpuResidentTensor<f32>> {
    let d_model = (n_heads * head_dim) as usize;
    let expected_size = (seq_len as usize) * d_model;

    // Validate input dimensions
    if q.len() != expected_size || k.len() != expected_size || v.len() != expected_size {
        return Err(crate::GpuError::InvalidParameter(format!(
            "Q/K/V size mismatch: expected {} (seq_len={}, d_model={})",
            expected_size, seq_len, d_model
        )));
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let batch = n_heads; // Each head is a "batch" item

    // Step 1: Convert interleaved -> batched for Q, K, V
    // [seq_len, n_heads * head_dim] -> [n_heads, seq_len, head_dim]
    let q_batched = interleaved_to_batched_all(ctx, q, seq_len, n_heads, head_dim)?;
    let k_batched = interleaved_to_batched_all(ctx, k, seq_len, n_heads, head_dim)?;
    let v_batched = interleaved_to_batched_all(ctx, v, seq_len, n_heads, head_dim)?;

    // Step 2: Transpose K for all heads
    // [n_heads, seq_len, head_dim] -> [n_heads, head_dim, seq_len]
    let kt_batched = batched_transpose_all(ctx, &k_batched, batch, seq_len, head_dim)?;

    // Step 3: Q @ K^T for all heads using BatchedGemmKernel
    // [n_heads, seq_len, head_dim] @ [n_heads, head_dim, seq_len] -> [n_heads, seq_len, seq_len]
    let scores = batched_gemm(ctx, &q_batched, &kt_batched, batch, seq_len, seq_len, head_dim)?;

    // Step 4: Scale all scores
    let total_scores = batch * seq_len * seq_len;
    let scaled_scores = batched_scale_all(ctx, &scores, scale, total_scores)?;

    // Step 5: Softmax for all heads (n_heads * seq_len rows of seq_len elements each)
    let attn = batched_softmax_all(ctx, &scaled_scores, batch * seq_len, seq_len)?;

    // Step 6: Attn @ V for all heads
    // [n_heads, seq_len, seq_len] @ [n_heads, seq_len, head_dim] -> [n_heads, seq_len, head_dim]
    let out_batched = batched_gemm(ctx, &attn, &v_batched, batch, seq_len, head_dim, seq_len)?;

    // Step 7: Convert batched -> interleaved
    // [n_heads, seq_len, head_dim] -> [seq_len, n_heads * head_dim]
    let output = batched_to_interleaved_all(ctx, &out_batched, seq_len, n_heads, head_dim)?;

    Ok(output)
}
