// ALB-091: GPU-resident gradient accumulation.
//
// Replaces CPU-side PerBlockGradientAccumulator when running with ga > 1.
// Eliminates the 24 × ga stream.synchronize() + D2H transfers per optimizer step
// that caused a 2.6x throughput regression (7.6K → 2.9K tok/s for 350M model).
//
// All accumulation happens on GPU via in-place element-wise add (ResidualAddKernel).
// Only ONE stream sync per optimizer step (not per micro-batch per block).
//
// VRAM cost: ~1.55 GB for 350M model (24 blocks × ~60 MB/block + 128 MB LM head).

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaContext, CudaStream, GpuBuffer};

#[cfg(feature = "cuda")]
use crate::autograd::cuda_forward::inplace_add_gpu;
#[cfg(feature = "cuda")]
use crate::transformer::cuda_block::CudaGradWorkspace;
#[cfg(feature = "cuda")]
use crate::transformer::TransformerConfig;

#[cfg(feature = "cuda")]
use super::grad_accumulator::BLOCK_GRAD_COMPONENTS;

#[cfg(feature = "cuda")]
fn gpu_err(e: impl std::fmt::Debug) -> crate::error::Error {
    crate::error::Error::ConfigError(format!("GPU error: {e:?}"))
}

/// GPU-resident gradient accumulation buffers for one transformer block.
#[cfg(feature = "cuda")]
pub struct GpuBlockGradAccum {
    /// 9 GPU buffers matching CudaGradWorkspace components:
    /// [w_q, w_k, w_v, w_o, gate, up, down, input_norm, post_attn_norm]
    pub components: Vec<GpuBuffer<f32>>,
    /// Pre-allocated host zero buffer (sized to largest component).
    zero_host: Vec<f32>,
}

#[cfg(feature = "cuda")]
impl GpuBlockGradAccum {
    fn new(ctx: &Arc<CudaContext>, sizes: &[usize; BLOCK_GRAD_COMPONENTS]) -> crate::Result<Self> {
        let mut components = Vec::with_capacity(BLOCK_GRAD_COMPONENTS);
        for &sz in sizes {
            components.push(GpuBuffer::new(ctx, sz).map_err(gpu_err)?);
        }
        let max_size = sizes.iter().copied().max().unwrap_or(0);
        Ok(Self { components, zero_host: vec![0.0f32; max_size] })
    }

    fn zero_all(&mut self) -> crate::Result<()> {
        for buf in &mut self.components {
            let n = buf.len();
            buf.copy_from_host(&self.zero_host[..n]).map_err(gpu_err)?;
        }
        Ok(())
    }
}

/// GPU-resident gradient accumulator for the full model.
///
/// Replaces `PerBlockGradientAccumulator` (CPU-side) when CUDA is available.
/// All gradient accumulation happens on GPU — zero D2H during micro-batch loop.
#[cfg(feature = "cuda")]
pub struct GpuGradientAccumulator {
    /// Per-block gradient accumulation buffers
    pub block_accums: Vec<GpuBlockGradAccum>,
    /// LM head gradient accumulator [vocab_size × hidden_size]
    pub lm_head_accum: GpuBuffer<f32>,
    /// Final norm gradient accumulator [hidden_size]
    pub final_norm_accum: GpuBuffer<f32>,
    /// Embedding gradient accumulator (CPU — embedding is CPU-side)
    pub embedding_accum: Vec<f32>,
    /// Number of accumulated micro-batches
    pub accumulated_count: usize,
    /// Component sizes per block (for iteration)
    pub block_component_sizes: [usize; BLOCK_GRAD_COMPONENTS],
    /// Pre-allocated host zero buffer for LM head zero
    lm_head_zero: Vec<f32>,
    /// Pre-allocated host zero buffer for final norm zero
    final_norm_zero: Vec<f32>,
}

#[cfg(feature = "cuda")]
impl GpuGradientAccumulator {
    /// Allocate GPU accumulation buffers matching the model architecture.
    ///
    /// VRAM cost: ~1.55 GB for 350M model (H=1024, I=4096, L=24, V=32768).
    pub fn new(ctx: &Arc<CudaContext>, config: &TransformerConfig) -> crate::Result<Self> {
        let h = config.hidden_size;
        let kv = config.num_kv_heads * config.head_dim();
        let i = config.intermediate_size;
        let v = config.vocab_size;

        let sizes =
            super::grad_accumulator::PerBlockGradientAccumulator::compute_block_sizes(h, kv, i);

        let mut block_accums = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            block_accums.push(GpuBlockGradAccum::new(ctx, &sizes)?);
        }

        let lm_head_accum = GpuBuffer::new(ctx, v * h).map_err(gpu_err)?;
        let final_norm_accum = GpuBuffer::new(ctx, h).map_err(gpu_err)?;
        let embedding_accum = vec![0.0f32; v * h];

        let total_vram_mb = (block_accums.len() as f64 * sizes.iter().sum::<usize>() as f64
            + (v * h) as f64
            + h as f64)
            * 4.0
            / (1024.0 * 1024.0);

        eprintln!(
            "  GPU gradient accumulation: {} blocks, {:.1} MB VRAM",
            config.num_hidden_layers, total_vram_mb,
        );

        let mut accum = Self {
            block_accums,
            lm_head_accum,
            final_norm_accum,
            embedding_accum,
            accumulated_count: 0,
            block_component_sizes: sizes,
            lm_head_zero: vec![0.0f32; v * h],
            final_norm_zero: vec![0.0f32; h],
        };

        // CRITICAL: GpuBuffer::new() does NOT zero VRAM (cuMemAlloc returns
        // uninitialized memory). Without this, the first accumulation window
        // adds real gradients to garbage → corrupted first optimizer step.
        accum.zero_all()?;

        Ok(accum)
    }

    /// Accumulate workspace gradients for a single block into GPU accum buffers.
    ///
    /// Uses in-place GPU add (ResidualAddKernel): accum[i] += workspace[i].
    /// No stream synchronization — fully asynchronous on the CUDA stream.
    pub fn accumulate_block(
        &mut self,
        workspace: &CudaGradWorkspace,
        block_idx: usize,
        stream: &CudaStream,
    ) -> crate::Result<()> {
        let accum = &mut self.block_accums[block_idx];
        let ws_bufs = workspace_buffers(workspace);

        for (comp_idx, (accum_buf, ws_buf)) in
            accum.components.iter_mut().zip(ws_bufs.iter()).enumerate()
        {
            let n = self.block_component_sizes[comp_idx] as u32;
            inplace_add_gpu(accum_buf, ws_buf, n, stream).map_err(gpu_err)?;
        }
        Ok(())
    }

    /// Accumulate non-block gradients (LM head + final norm) into GPU accum buffers.
    pub fn accumulate_nonblock(
        &mut self,
        lm_head_grad: &GpuBuffer<f32>,
        final_norm_grad: &GpuBuffer<f32>,
        stream: &CudaStream,
    ) -> crate::Result<()> {
        inplace_add_gpu(&mut self.lm_head_accum, lm_head_grad, lm_head_grad.len() as u32, stream)
            .map_err(gpu_err)?;
        inplace_add_gpu(
            &mut self.final_norm_accum,
            final_norm_grad,
            final_norm_grad.len() as u32,
            stream,
        )
        .map_err(gpu_err)?;
        Ok(())
    }

    /// Copy accumulated gradients back to workspace for optimizer step.
    ///
    /// Uses synchronous D2D copy. Must be called AFTER stream.synchronize().
    pub fn upload_to_workspace(
        &self,
        workspace: &mut CudaGradWorkspace,
        block_idx: usize,
    ) -> crate::Result<()> {
        let accum = &self.block_accums[block_idx];
        let ws_bufs = workspace_buffers_mut(workspace);

        for (ws_buf, accum_buf) in ws_bufs.into_iter().zip(accum.components.iter()) {
            ws_buf.copy_from_buffer(accum_buf).map_err(gpu_err)?;
        }
        Ok(())
    }

    /// Copy accumulated non-block gradients to the training buffers.
    pub fn upload_nonblock(
        &self,
        lm_head_grad: &mut GpuBuffer<f32>,
        final_norm_grad: &mut GpuBuffer<f32>,
    ) -> crate::Result<()> {
        lm_head_grad.copy_from_buffer(&self.lm_head_accum).map_err(gpu_err)?;
        final_norm_grad.copy_from_buffer(&self.final_norm_accum).map_err(gpu_err)?;
        Ok(())
    }

    /// Zero all accumulation buffers (call at start of each optimizer step).
    ///
    /// Uses H2D copy from pre-allocated zero buffers. Called once per optimizer step
    /// (not per micro-batch), so the H2D cost is negligible compared to savings.
    pub fn zero_all(&mut self) -> crate::Result<()> {
        for block in &mut self.block_accums {
            block.zero_all()?;
        }
        self.lm_head_accum.copy_from_host(&self.lm_head_zero).map_err(gpu_err)?;
        self.final_norm_accum.copy_from_host(&self.final_norm_zero).map_err(gpu_err)?;
        self.embedding_accum.iter_mut().for_each(|x| *x = 0.0);
        self.accumulated_count = 0;
        Ok(())
    }
}

/// Get references to workspace gradient buffers in component order.
#[cfg(feature = "cuda")]
fn workspace_buffers(ws: &CudaGradWorkspace) -> [&GpuBuffer<f32>; BLOCK_GRAD_COMPONENTS] {
    [
        &ws.grad_w_q,
        &ws.grad_w_k,
        &ws.grad_w_v,
        &ws.grad_w_o,
        &ws.grad_gate,
        &ws.grad_up,
        &ws.grad_down,
        &ws.grad_input_norm,
        &ws.grad_post_attn_norm,
    ]
}

/// Get mutable references to workspace gradient buffers in component order.
#[cfg(feature = "cuda")]
fn workspace_buffers_mut(
    ws: &mut CudaGradWorkspace,
) -> [&mut GpuBuffer<f32>; BLOCK_GRAD_COMPONENTS] {
    [
        &mut ws.grad_w_q,
        &mut ws.grad_w_k,
        &mut ws.grad_w_v,
        &mut ws.grad_w_o,
        &mut ws.grad_gate,
        &mut ws.grad_up,
        &mut ws.grad_down,
        &mut ws.grad_input_norm,
        &mut ws.grad_post_attn_norm,
    ]
}
