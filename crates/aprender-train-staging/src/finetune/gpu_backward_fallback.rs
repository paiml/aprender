//! CPU fallback for lm_head backward GEMM (PMAT-471).
//!
//! On VRAM-constrained GPUs (yoga 8GB), GPU embeddings don't fit after NF4 blocks.
//! Without GPU embeddings, lm_head backward GEMM silently fails and backward through
//! blocks is never called — the model cannot train.
//!
//! This module provides a CPU fallback: download grad_logits from GPU, multiply with
//! CPU embedding weights, upload grad_hidden to GPU. Slower but functional.

#[cfg(feature = "cuda")]
use crate::autograd::cuda_training::CudaTrainer;
#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaStream, GpuBuffer};

/// CPU fallback for lm_head backward: grad_hidden = grad_logits @ embed.
///
/// Downloads grad_logits[seq_len, vocab_size] from GPU, multiplies with CPU
/// embedding weights[vocab_size, hidden_size], uploads result to GPU.
///
/// # Cost
/// - D2H: seq_len × vocab_size × 4 bytes (~311 MB for seq=512, vocab=151936)
/// - CPU matmul: O(seq × vocab × hidden)
/// - H2D: seq_len × hidden_size × 4 bytes (~3 MB for seq=512, hidden=1536)
#[cfg(feature = "cuda")]
pub fn cpu_lmhead_backward(
    trainer: &CudaTrainer,
    logits_buf: &GpuBuffer<f32>,
    grad_hidden_buf: &mut GpuBuffer<f32>,
    embed_weights: &[f32],
    seq_len: usize,
    vocab_size: usize,
    hidden_size: usize,
    stream: &CudaStream,
) -> Option<()> {
    stream.synchronize().ok()?;

    let grad_logits = trainer.download(logits_buf).ok()?;

    // CPU matmul: grad_hidden[s,h] = Σ_v grad_logits[s,v] × embed[v,h]
    let mut grad_hidden = vec![0.0f32; seq_len * hidden_size];
    for s in 0..seq_len {
        for v in 0..vocab_size {
            let g = grad_logits[s * vocab_size + v];
            if g == 0.0 {
                continue;
            }
            let embed_row = &embed_weights[v * hidden_size..(v + 1) * hidden_size];
            let out_row = &mut grad_hidden[s * hidden_size..(s + 1) * hidden_size];
            for h in 0..hidden_size {
                out_row[h] += g * embed_row[h];
            }
        }
    }

    let gpu_grad = trainer.upload(&grad_hidden).ok()?;
    grad_hidden_buf.copy_from_buffer(&gpu_grad).ok()?;
    stream.synchronize().ok()?;

    eprintln!(
        "[CUDA] lm_head backward via CPU fallback (PMAT-471): \
         {:.1}MB D2H + CPU matmul + {:.1}MB H2D",
        (seq_len * vocab_size * 4) as f64 / 1e6,
        (seq_len * hidden_size * 4) as f64 / 1e6,
    );
    Some(())
}

/// Initialize FP16 weights for all NF4 blocks when FP16_GEMM=1 (PMAT-470).
///
/// Casts fp32 dequantized weights to fp16 on GPU using CastF32ToF16Kernel.
/// One-time cost at model initialization, amortized over all training steps.
#[cfg(feature = "cuda")]
pub fn init_fp16_weights(
    blocks: &mut [crate::transformer::CudaBlock],
    stream: &CudaStream,
) -> usize {
    use crate::transformer::CudaBlock;

    let mut ok = 0usize;
    for (i, block) in blocks.iter_mut().enumerate() {
        if let CudaBlock::Nf4(ref mut nf4) = block {
            match nf4.set_fp16_weights(stream) {
                Ok(()) => ok += 1,
                Err(e) => {
                    eprintln!("[FP16] Layer {i} cast failed: {e} — fp32 fallback");
                    break;
                }
            }
        }
    }
    if ok == blocks.len() {
        eprintln!("[FP16] All {ok} layers cast to fp16 — tensor core GEMM enabled");
    }
    ok
}

/// Pre-allocate cuBLAS workspace for CUDA graph capture (PMAT-063).
///
/// During graph capture, cuBLAS cannot allocate workspace dynamically.
/// Must be called BEFORE `stream.begin_capture()`. Returns the workspace
/// buffer that must be kept alive for the duration of graph use.
#[cfg(feature = "cuda")]
pub fn preallocate_cublas_workspace(trainer: &CudaTrainer) -> Option<GpuBuffer<f32>> {
    const WORKSPACE_BYTES: usize = 32 * 1024 * 1024; // 32 MB
    const WORKSPACE_ELEMS: usize = WORKSPACE_BYTES / 4;

    let ws = trainer.zeros(WORKSPACE_ELEMS).ok()?;
    let ws_ptr = ws.as_ptr();

    if let Err(e) = crate::autograd::cuda_forward::set_cublas_workspace(ws_ptr, WORKSPACE_BYTES) {
        eprintln!("[CUDA] cuBLAS workspace set failed: {e}");
        return None;
    }
    eprintln!("[CUDA] cuBLAS workspace pre-allocated: 32 MB");
    Some(ws)
}
