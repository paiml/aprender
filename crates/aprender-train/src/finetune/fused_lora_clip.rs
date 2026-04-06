//! PMAT-477: Fused LoRA gradient clipping — zero D2H sync for CUDA graph capture.
//!
//! Replaces per-layer synchronous `clip_gradients()` with ALB-078 fused pipeline:
//! 1. 6x `squared_sum_launch_into` (async, into contiguous partials buffer)
//! 2. 1x `clip_scale_reduce_cuda` (GPU-side norm + scale computation)
//! 3. 6x `gradient_clip_gpu_scale_cuda` (GPU-side scale read, no D2H)
//!
//! This eliminates 6 `stream.synchronize()` calls per backward step (168 per
//! backward pass across 28 layers), enabling CUDA graph capture of the backward loop.
//!
//! # Contract: cuda-graph-backward-v1.yaml
//!
//! - F-GRAPH-BWD-001: Loss trajectory matches ungraphed within 0.1
//! - F-GRAPH-BWD-002: Graph capture succeeds (no CUDA_ERROR)
//! - F-GRAPH-BWD-003: Throughput >= 1.10x ungraphed at batch=4

#[cfg(feature = "cuda")]
use crate::autograd::cuda_optim::{
    clip_scale_reduce_cuda, gradient_clip_gpu_scale_cuda, squared_sum_launch_into, FusedClipState,
};
#[cfg(feature = "cuda")]
use crate::transformer::cuda_block::CudaLoraGradWorkspace;
#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaContext, CudaStream, GpuBuffer};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::SquaredSumKernel;

/// Initialize a `FusedClipState` sized for the 6 LoRA gradient buffers.
///
/// Pre-allocates the contiguous partials buffer and scale output buffer.
/// Called once at training init, reused every backward step.
#[cfg(feature = "cuda")]
pub(crate) fn init_lora_fused_clip(
    ws: &CudaLoraGradWorkspace,
    ctx: &std::sync::Arc<CudaContext>,
) -> Option<FusedClipState> {
    let sizes: [u32; 6] = [
        ws.grad_lora_a_q.len() as u32,
        ws.grad_lora_b_q.len() as u32,
        ws.grad_lora_a_v.len() as u32,
        ws.grad_lora_b_v.len() as u32,
        ws.grad_input_norm.len() as u32,
        ws.grad_post_attn_norm.len() as u32,
    ];

    let mut offsets = [0u32; 9]; // FusedClipState uses [9] — pad unused
    let mut total = 0u32;
    for (i, &n) in sizes.iter().enumerate() {
        offsets[i] = total;
        let kernel = SquaredSumKernel::new(n);
        total += kernel.num_blocks();
    }

    let partials_buf = GpuBuffer::<f32>::new(ctx, total as usize).ok()?;
    let scale_buf = GpuBuffer::<f32>::new(ctx, 2).ok()?;

    Some(FusedClipState {
        partials_buf,
        scale_buf,
        offsets,
        num_blocks: [0; 9],
        total_partials: total,
    })
}

/// Apply fused gradient clipping to LoRA workspace — zero D2H sync.
///
/// Three-phase pipeline (all GPU-side, CUDA graph capturable):
/// 1. Launch squared-sum reductions for all 6 gradient buffers (async)
/// 2. Reduce partials and compute clip scale on GPU
/// 3. Apply clip scale from GPU memory to all 6 buffers
#[cfg(feature = "cuda")]
pub(crate) fn clip_lora_gradients_fused(
    ws: &mut CudaLoraGradWorkspace,
    max_norm: f32,
    state: &FusedClipState,
    stream: &CudaStream,
) {
    // Phase 1: Launch all 6 squared-sum reductions (async, no sync).
    let bufs: [&GpuBuffer<f32>; 6] = [
        &ws.grad_lora_a_q,
        &ws.grad_lora_b_q,
        &ws.grad_lora_a_v,
        &ws.grad_lora_b_v,
        &ws.grad_input_norm,
        &ws.grad_post_attn_norm,
    ];

    for (i, buf) in bufs.iter().enumerate() {
        let n = buf.len() as u32;
        if n == 0 {
            continue;
        }
        let output_ptr = state.partials_buf.as_ptr() + u64::from(state.offsets[i]) * 4;
        let _ = squared_sum_launch_into(buf, n, output_ptr, stream);
    }

    // Phase 2: Reduce all partials → clip_scale on GPU (no D2H).
    let _ = clip_scale_reduce_cuda(
        &state.partials_buf,
        state.total_partials,
        max_norm,
        &state.scale_buf,
        stream,
    );

    // Phase 3: Apply clip scale from GPU memory (no D2H).
    let scale_ptr = state.scale_buf.as_ptr();
    let bufs_mut: [&mut GpuBuffer<f32>; 6] = [
        &mut ws.grad_lora_a_q,
        &mut ws.grad_lora_b_q,
        &mut ws.grad_lora_a_v,
        &mut ws.grad_lora_b_v,
        &mut ws.grad_input_norm,
        &mut ws.grad_post_attn_norm,
    ];
    for buf in bufs_mut {
        let n = buf.len() as u32;
        if n == 0 {
            continue;
        }
        let _ = gradient_clip_gpu_scale_cuda(buf, scale_ptr, n, stream);
    }
}
